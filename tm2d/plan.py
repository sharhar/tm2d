import copy
import dataclasses
import inspect
import types
import vkdispatch as vd
import numpy as np
import tqdm
from typing import Optional

from .ctf.ctf import CTFParams

from .base_template import Template
from .base_comparator import Comparator
from .base_results import Results
from .param_set import ParamSet

class Plan:
    """Exhaustive search of templates against micrographs.

    Each submission of the command graph processes template_batch_size "slots"; per-pixel results are updated once
    per submission, so a larger template_batch_size means less result traffic per template (and more GPU memory:
    two (micrographs * template_batch_size, H, W) correlation buffers). Two ways to fill the slots:

      "per_slot": every slot holds any (rotation, CTF) combination of one pixel size (the parameter list is
          flattened), one projection per slot; full slots even with a single CTF value.
      "shared":   the slots of an instance share one rotation and hold K CTF values; one projection (or density
          sample) per instance, so it is cheaper when there are many CTF values.
      "auto" (default): "shared" if the number of CTF values is a multiple of template_batch_size, else "per_slot".
          The graph for the mode is built on first use (run) and rebuilt only if the mode changes.

    ctf_side: "template" applies the CTF to each template in its box (the slot modes above). "image" renders CTF-free
    projections and applies the CTF on the micrograph's Fourier grid inside the comparator (see
    ComparatorCrossCorrelation.compare_template, image_ctf): no box truncation or aliasing of the CTF, and each
    projection's forward FFT is shared by ctf_batch_size CTFs. A graph instance then holds template_batch_size
    rotations x ctf_batch_size CTFs (M * ctf_batch_size * template_batch_size correlation maps). "auto" (default):
    "image" whenever the image side is available (a comparator with image_ctf and fused norms, a template with a
    disable_ctf switch, rotation_sharing "auto"), for any number of CTF values, so a given parameter scores the same
    whether it is searched alone or with others; else "template". Measured (log entries 26-28): with several defoci
    the image side was 1.04-2.6x faster in every tested configuration; with one defocus 1.13x faster at 256^2 and
    1.4% slower at 4096^2.
    ctf_batch_size=None: chosen per ParamSet, the largest divisor of the number of CTF values for which the
    M * ctf_batch_size * template_batch_size correlation maps fit in one device buffer (no padded CTF slots). kernels_per_pass (image side): the
    comparator's fused column pass handles at most this many (micrograph, CTF) kernels per shader, CTF-major, reusing
    each CTF across micrographs (None = all in one pass)."""
    template: Template
    comparator: Comparator
    results: Results
    rotation: Optional[np.ndarray]
    pixel_size: Optional[float]
    ctf_params: CTFParams
    template_batch_size: int
    enable_rotation_weights: bool

    _status_bar: Optional[tqdm.tqdm]

    def __init__(self,
                 template: Template,
                 comparator: Comparator,
                 results: Results,
                 rotation: Optional[np.ndarray] = None,
                 pixel_size: Optional[float] = None,
                 ctf_params: Optional[CTFParams] = None,
                 template_batch_size: int = 8,
                 output_radius: Optional[int] = None,
                 enable_rotation_weights: bool = False,
                 rotation_sharing: str = "auto",
                 ctf_side: str = "auto",
                 ctf_batch_size: Optional[int] = None,
                 kernels_per_pass: Optional[int] = 8):

        if rotation_sharing not in ("auto", "per_slot", "shared"):
            raise ValueError(f"rotation_sharing must be 'auto', 'per_slot' or 'shared', not {rotation_sharing!r}")
        self.template_batch_size = int(template_batch_size)
        if ctf_side not in ("auto", "template", "image"):
            raise ValueError(f"ctf_side must be 'auto', 'template' or 'image', not {ctf_side!r}")
        self.ctf_side = ctf_side
        self._ctf_batch_size_arg = None if ctf_batch_size is None else int(ctf_batch_size)
        self.ctf_batch_size = self._ctf_batch_size_arg      # the D of the built image-side graph
        self.kernels_per_pass = kernels_per_pass

        if ctf_params is None:
            ctf_params = CTFParams()

        self.template = template
        self.comparator = comparator
        self.results = results
        self.results.set_template_shape(self.template.get_shape())
        self.rotation = rotation
        self.pixel_size = pixel_size
        self.ctf_params = ctf_params
        self.enable_rotation_weights = enable_rotation_weights
        self.output_radius = output_radius
        self.rotation_sharing = rotation_sharing

        self._status_bar = None
        self.mode = None
        self.cmd_graph = None
        if ctf_side == "image":
            if self.ctf_batch_size is not None:
                self._build("image")
        elif rotation_sharing != "auto":
            self._build(rotation_sharing)       # "auto" then stays on the template side (image_side_available)

    def _build(self, mode: str):
        """Record the command graph for slot mode `mode` ("per_slot" or "shared")."""
        self.mode = None
        self.cmd_graph = vd.CommandGraph()

        prev_graph = vd.set_global_graph(self.cmd_graph)
        try:          # restore the previous global graph even if recording fails (e.g. a buffer is too large)
            self._record(mode)
        finally:
            vd.set_global_graph(prev_graph)
        self.mode = mode

    def _static_ctf_params(self) -> CTFParams:
        """The plan's CTF parameters with searched (None) fields set to CTFParams defaults: for rendering CTF-free
        templates (only HT, for the interaction constant, is used then)."""
        p = copy.copy(self.ctf_params)
        defaults = CTFParams()
        for f in dataclasses.fields(p):
            if getattr(p, f.name) is None:
                setattr(p, f.name, getattr(defaults, f.name))
        return p

    def _make_ctf_free_template(self, rotations, pixel_size, K):
        """make_template with the template's disable_ctf switched on (restored afterwards): projections only."""
        had_attr = hasattr(self.template, "disable_ctf")
        old = getattr(self.template, "disable_ctf", None)
        self.template.disable_ctf = True
        try:
            return self.template.make_template(rotations, pixel_size, template_count=K, cmd_graph=self.cmd_graph,
                                               ctf_params=self._static_ctf_params())
        finally:
            if had_attr:
                self.template.disable_ctf = old
            else:
                del self.template.disable_ctf

    def _record_image(self):
        """Graph for ctf_side="image": K rotations per instance (one CTF-free projection each) x D CTFs applied in the
        comparator. Slot i = d * K + k (CTF d, rotation k)."""
        K, D = self.template_batch_size, self.ctf_batch_size
        if self.rotation is not None:
            rotations = self.template.get_rotation_matrices(self.rotation)
        else:
            rotations = [self.cmd_graph.bind_var(f"rotation_matrix{k}") for k in range(K)]
        pixel_size = self.cmd_graph.bind_var("pixel_size") if self.pixel_size is None else self.pixel_size

        self.template_buffer = self._make_ctf_free_template(rotations, pixel_size, K)

        image_ctf = types.SimpleNamespace(ctf_params=self.ctf_params, count=D, pixel_size=pixel_size,
                                          args=self.ctf_params.get_args(self.cmd_graph, D),
                                          static_params=self.ctf_params,
                                          kernels_per_pass=self.kernels_per_pass)
        mic_buffer = getattr(self.comparator, "micrographs_buffer", None)
        trim = (mic_buffer is not None and hasattr(self.results, "edges_excluded") and self.output_radius is None
                and self.results.edges_excluded(tuple(int(n) for n in mic_buffer.real_shape[1:3])))
        if "image_ctf" not in inspect.signature(self.comparator.compare_template).parameters:
            raise ValueError(f"{type(self.comparator).__name__} does not support ctf_side='image'")
        self.comparison_buffer = self.comparator.compare_template(
            self.template_buffer, output_radius=self.output_radius, trim_to_valid=trim, image_ctf=image_ctf)
        self.results.set_score_norms(getattr(self.comparator, "score_norms", None),
                                     getattr(self.comparator, "score_norm_inv_n", 1.0))
        if not self.enable_rotation_weights:
            weights = 1
        else:
            weights = [self.cmd_graph.bind_var(f"rotation_weight{k}") for k in range(K)] * D
        self.results.check_comparison(
            self.comparison_buffer,
            weights,
            *[self.cmd_graph.bind_var(f"index{i}") for i in range(D * K)]
        )

    def _record(self, mode: str):
        if mode == "image":
            self._record_image()
            return

        K = self.template_batch_size
        if self.rotation is not None:
            rotations = self.template.get_rotation_matrices(self.rotation)
        elif mode == "per_slot":
            rotations = [self.cmd_graph.bind_var(f"rotation_matrix{k}") for k in range(K)]   # one per slot
        else:
            rotations = self.cmd_graph.bind_var("rotation_matrix")                            # shared by the slots

        pixel_size = self.cmd_graph.bind_var("pixel_size") if self.pixel_size is None else self.pixel_size
        if getattr(self.comparator, "prefiltered", False):
            # the CTF is already in the (prefiltered) data: CTF-free projections
            self.template_buffer = self._make_ctf_free_template(rotations, pixel_size, K)
        else:
            self.template_buffer = self.template.make_template(
                rotations,
                pixel_size,
                template_count=K,
                cmd_graph=self.cmd_graph,
                ctf_params=self.ctf_params
            )

        # with edge positions excluded, the comparator need not compute the rows / columns that hold none
        mic_buffer = getattr(self.comparator, "micrographs_buffer", None)
        trim = (mic_buffer is not None and hasattr(self.results, "edges_excluded") and self.output_radius is None
                and self.results.edges_excluded(tuple(int(n) for n in mic_buffer.real_shape[1:3])))
        # comparators without trim support (no trim_to_valid parameter) are called as before
        if "trim_to_valid" in inspect.signature(self.comparator.compare_template).parameters:
            self.comparison_buffer = self.comparator.compare_template(
                self.template_buffer, output_radius=self.output_radius, trim_to_valid=trim)
        else:
            self.comparison_buffer = self.comparator.compare_template(
                self.template_buffer, output_radius=self.output_radius)

        # factors the comparator leaves to the results stage (fused template norms), if any
        self.results.set_score_norms(getattr(self.comparator, "score_norms", None),
                                     getattr(self.comparator, "score_norm_inv_n", 1.0))

        if not self.enable_rotation_weights:
            weights = 1
        elif mode == "per_slot":
            weights = [self.cmd_graph.bind_var(f"rotation_weight{k}") for k in range(K)]
        else:
            weights = [self.cmd_graph.bind_var("rotation_weight")] * K
        self.results.check_comparison(
            self.comparison_buffer,
            weights,
            *[self.cmd_graph.bind_var(f"index{k}") for k in range(K)]
        )


    def set_data(self, data: np.ndarray, weights=None):
        """weights: optional whitening weights for the comparator (see ComparatorCrossCorrelation.set_data)."""
        self.comparator.set_data(data, weights=weights)

    def reset(self):
        self.results.reset()

    def prepare(self, params: ParamSet):
        """Build (compile) the command graph for the slot mode `params` needs, if not built yet. run() calls it; call
        it beforehand to keep graph compilation out of a timed run."""
        mode = self.choose_mode(params)
        if mode == "image":
            D = self._ctf_batch_size_arg or self._auto_ctf_batch_size(params.get_ctf_count())
            if mode != self.mode or D != self.ctf_batch_size:
                self.ctf_batch_size = D
                self._build(mode)
        elif mode != self.mode:
            self._build(mode)

    def image_side_available(self) -> bool:
        """Whether ctf_side="auto" may use the image side: the comparator applies image-side CTFs with fused norms,
        the template can render CTF-free (a disable_ctf switch, so a template that applies its own CTF is never
        filtered twice), and no template-side slot mode was forced."""
        return ("image_ctf" in inspect.signature(self.comparator.compare_template).parameters
                and not getattr(self.comparator, "prefiltered", False)
                and getattr(self.comparator, "norm_method", "fused") == "fused"
                and hasattr(self.template, "disable_ctf")
                and self.rotation_sharing == "auto")

    def _auto_ctf_batch_size(self, C: int) -> int:
        """Largest divisor D of C with M * D * K correlation maps (rfft, complex64) in one device buffer."""
        from .limits import max_buffer_bytes
        mic = getattr(self.comparator, "micrographs_buffer", None)
        if mic is None:
            return int(C)
        M, H, W = (int(n) for n in mic.real_shape)
        per_map = H * (W // 2 + 1) * 8
        fit = max(1, max_buffer_bytes() // (per_map * M * self.template_batch_size))
        return max(d for d in range(1, int(C) + 1) if C % d == 0 and d <= fit)

    def choose_mode(self, params: ParamSet) -> str:
        if getattr(self.comparator, "prefiltered", False):
            # prefiltered data (tm2d.prefilter): each micrograph is already filtered by one CTF
            if params.get_ctf_count() != 1:
                raise ValueError("a prefiltered comparator takes a ParamSet with one CTF value (the CTF is in the data: "
                                 "one micrograph per defocus)")
            if not hasattr(self.template, "disable_ctf"):
                raise ValueError("a prefiltered comparator needs a template with a disable_ctf switch (CTF-free "
                                 "projections)")
            if self.ctf_side == "image":
                raise ValueError("ctf_side='image' cannot be used with a prefiltered comparator")
            return self.rotation_sharing if self.rotation_sharing != "auto" else "per_slot"
        if self.ctf_side == "image":
            return "image"
        if self.ctf_side == "auto" and self.image_side_available():
            return "image"
        if self.rotation_sharing != "auto":
            return self.rotation_sharing
        C, K = params.get_ctf_count(), self.template_batch_size
        return "shared" if C >= K and C % K == 0 else "per_slot"

    def _submit_slots(self, params: ParamSet, pixel_index: int, p: np.ndarray, r: np.ndarray, c: np.ndarray,
                      rotation_matrices: Optional[np.ndarray]):
        """One submission: parameter indices p (rotation r, CTF c, all at pixel size pixel_index, -1 = empty slot),
        a multiple of template_batch_size long, laid into instances of template_batch_size slots. In "shared" mode
        the slots of an instance all have the same rotation."""
        K = self.template_batch_size
        n_inst = len(p) // K
        p, r, c = p.reshape(n_inst, K), r.reshape(n_inst, K), c.reshape(n_inst, K)
        shared = self.mode == "shared"

        if rotation_matrices is not None and shared:
            self.cmd_graph.set_var("rotation_matrix", rotation_matrices[r[:, 0]])
        if self.enable_rotation_weights and shared:
            self.cmd_graph.set_var("rotation_weight", params.rotations_weights[r[:, 0]].astype(np.float32))
        for k in range(K):
            if rotation_matrices is not None and not shared:
                self.cmd_graph.set_var(f"rotation_matrix{k}", rotation_matrices[r[:, k]])
            for ii, field in enumerate(params.ctf_set.field_names if not getattr(self.comparator, "prefiltered", False) else []):
                self.cmd_graph.set_var(f"{field}_{k}",      # (prefiltered: CTF-free templates bind no CTF variables)
                                       params.ctf_set.combinations_array[c[:, k], ii].astype(np.float32))
            if self.enable_rotation_weights and not shared:
                weights = params.rotations_weights[r[:, k]].astype(np.float32)
                weights[p[:, k] < 0] = 0.0
                self.cmd_graph.set_var(f"rotation_weight{k}", weights)
            self.cmd_graph.set_var(f"index{k}", p[:, k].astype(np.int32))

        if params.pixel_sizes is not None and self.pixel_size is None:
            self.cmd_graph.set_var("pixel_size", np.full(n_inst, params.pixel_sizes[pixel_index], np.float32))

        self.cmd_graph.submit_any(n_inst)

        if self._status_bar is not None:
            self._status_bar.update(int(np.sum(p >= 0)))

    def _submit_image(self, params: ParamSet, pixel_index: int, r: np.ndarray, c: np.ndarray,
                      rotation_matrices: Optional[np.ndarray]):
        """One submission in "image" mode: r (n_inst, K) rotation indices and c (n_inst, D) CTF indices, -1 = empty."""
        K, D = self.template_batch_size, self.ctf_batch_size
        n_inst = r.shape[0]
        S, C = params.get_pixel_size_count(), params.get_ctf_count()
        rr, cc = np.maximum(r, 0), np.maximum(c, 0)
        for k in range(K):
            if rotation_matrices is not None:
                self.cmd_graph.set_var(f"rotation_matrix{k}", rotation_matrices[rr[:, k]])
            if self.enable_rotation_weights:
                weights = params.rotations_weights[rr[:, k]].astype(np.float32)
                weights[r[:, k] < 0] = 0.0
                self.cmd_graph.set_var(f"rotation_weight{k}", weights)
        for d in range(D):
            for ii, field in enumerate(params.ctf_set.field_names):
                self.cmd_graph.set_var(f"{field}_{d}", params.ctf_set.combinations_array[cc[:, d], ii].astype(np.float32))
        for d in range(D):
            for k in range(K):
                p = np.where((r[:, k] >= 0) & (c[:, d] >= 0), (r[:, k] * S + pixel_index) * C + c[:, d], -1)
                self.cmd_graph.set_var(f"index{d * K + k}", p.astype(np.int32))
        if params.pixel_sizes is not None and self.pixel_size is None:
            self.cmd_graph.set_var("pixel_size", np.full(n_inst, params.pixel_sizes[pixel_index], np.float32))
        self.cmd_graph.submit_any(n_inst)
        if self._status_bar is not None:
            self._status_bar.update(int(np.sum((r[:, :, None] >= 0) & (c[:, None, :] >= 0))))

    def _run_image(self, params: ParamSet, rotation_matrices, batch_size: int):
        K, D = self.template_batch_size, self.ctf_batch_size
        R, S, C = params.get_rotation_count(), params.get_pixel_size_count(), params.get_ctf_count()
        nr, nc = -(-R // K), -(-C // D)
        rot = np.arange(nr * K).reshape(nr, K)
        rot[rot >= R] = -1
        ctf = np.arange(nc * D).reshape(nc, D)
        ctf[ctf >= C] = -1
        # instance (rotation group j, CTF group g); CTF groups inner, so the rotation groups run in order
        r_all = np.repeat(rot, nc, axis=0)
        c_all = np.tile(ctf, (nr, 1))
        for s in range(S):
            for start in range(0, nr * nc, int(batch_size)):
                sl = slice(start, start + int(batch_size))
                self._submit_image(params, s, r_all[sl], c_all[sl], rotation_matrices)

    def run(self,
            params: ParamSet,
            enable_progress_bar: bool = False,
            batch_size: int = 32,
            compile_results: bool = True):
        """batch_size: graph instances per submission (each instance = template_batch_size slots)."""

        if params.rotations_weights is None and self.enable_rotation_weights:
            raise ValueError("enable_rotation_weights is True, but no rotation weights provided in params.")

        self.prepare(params)

        K = self.template_batch_size
        R, S, C = params.get_rotation_count(), params.get_pixel_size_count(), params.get_ctf_count()
        rotation_matrices = None
        if params.rotations is not None and self.rotation is None:
            rotation_matrices = self.template.get_rotation_matrices(np.asarray(params.rotations, dtype=np.float32))

        if enable_progress_bar:
            self._status_bar = tqdm.tqdm(total=params.get_total_count(), dynamic_ncols=True)

        # parameter index p = (rotation * S + pixel) * C + ctf (ParamSet.get_values_at_index). A submission never
        # mixes pixel sizes (one pixel size per graph instance).
        if self.mode == "image":
            self._run_image(params, rotation_matrices, batch_size)
            S = 0                                     # skip the template-side loop below
        if self.mode == "per_slot":
            rot = np.repeat(np.arange(R), C)          # slots filled freely from the flattened list
            ctf = np.tile(np.arange(C), R)
        elif self.mode == "shared":
            Cp = -(-C // K) * K                       # each rotation's CTF list padded to a multiple of K
            rot = np.repeat(np.arange(R), Cp)
            ctf = np.tile(np.arange(Cp), R)
        per_submit = K * int(batch_size)
        for s in range(S):
            p_all = np.where(ctf < C, (rot * S + s) * C + ctf, -1)
            for start in range(0, len(p_all), per_submit):
                sl = slice(start, start + per_submit)
                p, r, c = p_all[sl], rot[sl], np.minimum(ctf[sl], C - 1)
                pad = (-len(p)) % K
                if pad:
                    p = np.concatenate([p, -np.ones(pad, np.int64)])
                    r = np.concatenate([r, np.zeros(pad, np.int64)])
                    c = np.concatenate([c, np.zeros(pad, np.int64)])
                self._submit_slots(params, s, p, r, c, rotation_matrices)

        if compile_results:
            # with rotation weights the per-pixel moments are weighted sums: normalize by the total weight
            total_weight = None
            if self.enable_rotation_weights:
                total_weight = float(np.sum(params.rotations_weights)) * params.get_ctf_count() * params.get_pixel_size_count()
            self.results.compile_results(params.get_total_count(), total_weight=total_weight)

        if self._status_bar is not None:
            self._status_bar.close()
            self._status_bar = None
