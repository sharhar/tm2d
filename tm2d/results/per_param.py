import vkdispatch as vd
import vkdispatch.codegen as vc

import numpy as np

from ..plan import Results, ParamSet
from ..base_results import valid_center_range

# The reductions below carry (max, sum, sum of squares) in one vec3, and vkdispatch starts every reduction from a
# single scalar identity (0 here) broadcast to all three components. A plain max would then never report less than
# 0, which a small output_radius box can need. The max is therefore carried as f(v) = v + sqrt(v^2 + 1), which is
# > 0 and increasing in v, so 0 (identity, or a pixel outside the region) never wins; decode_max inverts it.
def encode_max(v):
    # v + sqrt(v^2 + 1) cancels catastrophically for v << 0 (maps reach the reduction unscaled with fused norms, so
    # |v| can be large); 1 / (sqrt(v^2 + 1) - v) is the same value without cancellation
    r = vc.sqrt(v * v + 1.0)
    return vc.select(v >= 0.0, v + r, 1.0 / (r - v)) if hasattr(vc, "select") else _encode_branch(v, r)

def _encode_branch(v, r):
    out = vc.new_float_register(0.0)
    with vc.if_block(v >= 0.0):
        out[:] = v + r
    with vc.else_block():
        out[:] = 1.0 / (r - v)
    return out

def decode_max(f):
    return 0.5 * (f - 1.0 / f)

def _in_valid(coord_y, coord_x, height, width, bounds, shifts):
    """True if raw correlation position (row coord_y, real column coord_x) is a valid position: its centered map index
    ((y + H//2) % H, (x + W//2) % W) lies in bounds = (ylo, yhi, xlo, xhi) (see valid_center_range)."""
    Y = ((coord_y + shifts.x) % height).to_register()
    X = ((coord_x + shifts.y) % width).to_register()
    return vc.all(Y >= bounds.x, Y <= bounds.y, X >= bounds.z, X <= bounds.w)

@vd.map
def reduction_map_radius(buf: vc.Buff[vc.c64], output_radius: vc.Const[vc.i32], bounds: vc.Const[vc.iv4],
                         shifts: vc.Const[vc.iv2]) -> vc.v3:
    """Moments and max over the 2r x 2r real pixels the comparator writes with output_radius = r (rows y < r or
    y >= H - r, real columns x < r or x >= W - r; complex column z holds real columns 2z and 2z + 1), restricted to
    valid positions (bounds)."""
    ind = vd.reduce.mapped_io_index()

    coord3d = vc.ravel_index(ind, buf.shape).to_register()

    result = vc.new_vec3_register(0, 0, 0)

    height = buf.shape.y.to_dtype(vc.i32).to_register()
    width = ((buf.shape.z - 1) * 2).to_dtype(vc.i32).to_register()
    y = coord3d.y.to_dtype(vc.i32).to_register()
    x0 = (coord3d.z.to_dtype(vc.i32) * 2).to_register()
    x1 = (x0 + 1).to_register()

    with vc.if_block(vc.any(y < output_radius, y >= height - output_radius)):
        val = buf[ind].to_register()
        with vc.if_block(vc.all(x0 < width, vc.any(x0 < output_radius, x0 >= width - output_radius),
                                _in_valid(y, x0, height, width, bounds, shifts))):
            result.x = encode_max(val.real)
            result.y = val.real
            result.z = val.real * val.real
        with vc.if_block(vc.all(x1 < width, vc.any(x1 < output_radius, x1 >= width - output_radius),
                                _in_valid(y, x1, height, width, bounds, shifts))):
            result.x = vc.max(result.x, encode_max(val.imag))
            result.y = result.y + val.imag
            result.z = result.z + val.imag * val.imag

    return result

@vd.reduce.reduce(0, axes=[1, 2], mapping_function=reduction_map_radius)
def calculate_sums_and_max_radius(a: vc.Const[vc.v3], b: vc.Const[vc.v3]) -> vc.v3:
    return vc.new_vec3_register(
        vc.max(a.x, b.x),
        a.y + b.y,
        a.z + b.z
    )

@vd.map
def reduction_map(buf: vc.Buff[vc.c64], bounds: vc.Const[vc.iv4], shifts: vc.Const[vc.iv2]) -> vc.v3:
    """Moments and max over the valid positions (bounds) of the full correlation map."""
    ind = vd.reduce.mapped_io_index()
    coord3d = vc.ravel_index(ind, buf.shape).to_register()

    result = vc.new_vec3_register(0, 0, 0)

    height = buf.shape.y.to_dtype(vc.i32).to_register()
    width = ((buf.shape.z - 1) * 2).to_dtype(vc.i32).to_register()
    y = coord3d.y.to_dtype(vc.i32).to_register()
    x0 = (coord3d.z.to_dtype(vc.i32) * 2).to_register()

    with vc.if_block(x0 < width):
        val = buf[ind].to_register()
        with vc.if_block(_in_valid(y, x0, height, width, bounds, shifts)):
            result.x = encode_max(val.real)
            result.y = val.real
            result.z = val.real * val.real
        with vc.if_block(_in_valid(y, x0 + 1, height, width, bounds, shifts)):
            result.x = vc.max(result.x, encode_max(val.imag))
            result.y = result.y + val.imag
            result.z = result.z + val.imag * val.imag

    return result

@vd.reduce.reduce(0, axes=[1, 2], mapping_function=reduction_map)
def calculate_sums_and_max(a: vc.Const[vc.v3], b: vc.Const[vc.v3]) -> vc.v3:
    return vc.new_vec3_register(
        vc.max(a.x, b.x),
        a.y + b.y,
        a.z + b.z
    )


class ResultsParam(Results):
    best_values_buffer: vd.Buffer
    best_mip_values_buffer: vd.Buffer
    compiled: bool
    compiled_best_values: np.ndarray
    compiled_best_mip_values: np.ndarray
    output_radius: int

    def __init__(self, batch_count: int, total_indices: int, output_radius: int = None, exclude_edges="auto") -> None:
        """exclude_edges: True = the max and the moments use only positions where the template lies fully inside
        the micrograph (valid positions only; elsewhere the circular correlation wraps the template onto the
        opposite edge); False = every position; "auto" (default) = True for micrograph searches, False for particle
        boxes (template as large as the image, or output_radius set); see Results.edges_excluded."""
        self.exclude_edges_setting = self._check_edge_setting(exclude_edges)
        self.best_values_buffer = vd.Buffer((batch_count, total_indices, ), vd.float32)
        self.best_mip_values_buffer = vd.Buffer((batch_count, total_indices, ), vd.float32)
        self.output_radius = output_radius
        self.compiled = False
        self.reset()

    def reset(self):
        self.best_values_buffer.write(
            (np.ones(shape=self.best_values_buffer.shape) * -1000000).astype(np.float32)
        )
        self.best_mip_values_buffer.write(
            (np.ones(shape=self.best_mip_values_buffer.shape) * -1000000).astype(np.float32)
        )

    def check_comparison(self, comparison_buffer: vd.RFFTBuffer, rotation_weights: vc.Var[vc.f32], *indices: vc.Var[vc.i32]):

        template_count = comparison_buffer.shape[0] // self.best_values_buffer.shape[0]

        assert comparison_buffer.shape[0] % self.best_values_buffer.shape[0] == 0, \
            "The comparison buffer shape must be divisible by the best values buffer shape."

        assert len(indices) == template_count, \
            f"The number of indices ({len(indices)}) must match the number of templates"

        H = int(comparison_buffer.shape[1])
        W = int((comparison_buffer.shape[2] - 1) * 2)
        if self.edges_excluded((H, W)):
            ylo, yhi = valid_center_range(H, self.template_shape[0])
            xlo, xhi = valid_center_range(W, self.template_shape[1])
        else:
            ylo, yhi, xlo, xhi = 0, H - 1, 0, W - 1
        bounds = np.array([ylo, yhi, xlo, xhi], dtype=np.int32)
        shifts = np.array([H // 2, W // 2], dtype=np.int32)
        # the mean and variance divide by the number of pixels actually reduced: the valid positions, within the
        # 2r x 2r output box if output_radius is set (before: H * W, which made the mean ~0 and inflated z)
        raw_y, raw_x = np.arange(H), np.arange(W)
        if self.output_radius is not None:
            r = self.output_radius
            raw_y = raw_y[(raw_y < r) | (raw_y >= H - r)]
            raw_x = raw_x[(raw_x < r) | (raw_x >= W - r)]
        Y, X = (raw_y + shifts[0]) % H, (raw_x + shifts[1]) % W
        pixel_count = int(np.sum((Y >= ylo) & (Y <= yhi))) * int(np.sum((X >= xlo) & (X <= xhi)))
        if pixel_count == 0:
            raise ValueError("no valid positions: the template does not fit inside the micrograph / output box")

        # rotation weights do not enter the per-parameter scores; a score scale (fused norms) is applied to the
        # reduced max, sum and sum of squares before the z-score
        scale_types = [vc.Buff[vc.f32]] if self.score_norms is not None else []
        inv_n = self.score_norm_inv_n

        @vd.shader(
            exec_size=self.best_values_buffer.shape[0],
            arg_type_annotations=[
                vc.Buff[vc.f32],  # zscore_buff
                vc.Buff[vc.f32],  # mip_buff
                vc.Buff[vc.v3],  # reduced_values_buff
            ] + scale_types + [vc.Var[vc.i32]] * len(indices)
        )
        def update_best_value(
            zscore_buff: vc.Buff[vc.f32],
            mip_buff: vc.Buff[vc.f32],
            reduced_values_buff: vc.Buff[vc.v3],
            *rest):
            scale_buff = rest[0] if scale_types else None
            indices = rest[len(scale_types):]

            ind = vc.global_invocation_id().x.to_dtype(vc.i32).to_register()

            reduced_values = vc.new_vec3_register()

            for i in range(template_count):
                output_index = ind * self.best_values_buffer.shape[1] + indices[i]
                input_index = ind * template_count + i

                reduced_values[:] = reduced_values_buff[input_index]
                reduced_values.x = decode_max(reduced_values.x)   # the max is carried encoded (see encode_max)

                with vc.if_block(indices[i] >= 0):
                    if scale_buff is not None:
                        # scale max, mean and variance first, so the 1e-6 regularizer acts on the normalized map
                        # as in v0.3.0 (the unscaled map's variance is ~ ||t||_w^2, which can be ~1e-6 itself)
                        s = (1.0 / vc.sqrt(scale_buff[input_index] * inv_n)).to_register()
                        max_val = (reduced_values.x * s).to_register()
                        mean_val = (reduced_values.y * s / float(pixel_count)).to_register()
                        var_val = vc.max(reduced_values.z * s * s / float(pixel_count) - mean_val * mean_val, 0.0).to_register()
                    else:
                        max_val = reduced_values.x.to_register()
                        mean_val = (reduced_values.y / float(pixel_count)).to_register()
                        var_val = vc.max(reduced_values.z / float(pixel_count) - mean_val * mean_val, 0.0).to_register()
                    z_score = ((max_val - mean_val) / vc.sqrt(var_val + 1e-6)).to_register()

                    with vc.if_block(z_score > zscore_buff[output_index]):
                        zscore_buff[output_index] = z_score
                        mip_buff[output_index] = max_val

        if self.output_radius is not None:
            reduced_values_buff = calculate_sums_and_max_radius(comparison_buffer, self.output_radius, bounds, shifts)
        else:
            reduced_values_buff = calculate_sums_and_max(comparison_buffer, bounds, shifts)

        update_best_value(
            self.best_values_buffer,
            self.best_mip_values_buffer,
            reduced_values_buff,
            *([self.score_norms] if self.score_norms is not None else []),
            *indices
        )

    def compile_results(self, template_count: int, total_weight: float = None):
        _ = template_count, total_weight

        # We do an element wise max reduction on the buffer because
        # each of the devices will have only written to a portion of the buffer
        # (which is unique to each device). So this will give us the full combined result.
        all_best_values = np.array(self.best_values_buffer.read())
        all_best_mips = np.array(self.best_mip_values_buffer.read())

        # Select the device slot that produced the winning z-score, then read both
        # z-score and matching MIP from that same slot.
        max_indices = np.argmax(all_best_values, axis=0)
        gather_indices = np.expand_dims(max_indices, axis=0)
        self.compiled_best_values = np.take_along_axis(all_best_values, gather_indices, axis=0)[0]
        self.compiled_best_mip_values = np.take_along_axis(all_best_mips, gather_indices, axis=0)[0]

        self.compiled = True

    def get_zscore_list(self, params: ParamSet):
        flat_zscore = self.compiled_best_values[:, :params.get_total_count()]
        zscore_shape = params.get_tensor_shape(flat_zscore.shape[0])[:-1]
        return flat_zscore.reshape(zscore_shape)

    def get_mip_list(self, params: ParamSet):
        flat_mip = self.compiled_best_mip_values[:, :params.get_total_count()]
        mip_shape = params.get_tensor_shape(flat_mip.shape[0])[:-1]
        return flat_mip.reshape(mip_shape)

