"""Fused column pass with the template norms folded in (zipFFT-style fusion, written for vkdispatch).

The shader is vkdispatch's convolution pipeline (vd.fft.convolve: read input -> forward FFT -> per kernel: map ->
inverse FFT -> write), with one addition: while the kernel map multiplies each template spectrum value T(s) by the
image, it also accumulates factor(s) * w(s) * |T(s)|^2 in a register, and after the map every thread writes its sum
into a small partials buffer (one float per thread and kernel). A tiny reduction then gives
||t||_w^2 * N = sum_s w(s) |T(s)|^2 over the full micrograph grid, for every (micrograph, template), with no extra
FFT. vkdispatch 0.1.0 internals are used (fft_context, io manager, grid.tid); they are pinned by tm2d's dependency.
"""
from typing import Tuple

import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.fft.shader_factories import set_global_kernel_index


class FusedNormState:
    """Shader-generation state shared between the factory and the kernel map: the per-thread accumulator register
    and the partials buffer argument."""

    def __init__(self):
        self.acc = None
        self.partials = None

    def begin(self):
        self.acc[:] = 0.0

    def record(self, io_index, value):
        self.acc[:] = self.acc + value


def make_fused_norm_convolution_shader(buffer_shape: Tuple, kernel_map: vd.MappingFunction, kernel_num: int,
                                       axis: int, input_map: vd.MappingFunction, output_map: vd.MappingFunction,
                                       state: FusedNormState, kernel_offset: int, template_count: int,
                                       columns: int):
    """Returns (shader, partials_per_row). Shader arguments, in order: output_map's, input_map's, kernel_map's
    (kernel_map must call state.record(...) once per register and expose the partials buffer as state.partials).
    The partials buffer is (kernel_num * template_count, partials_per_row): row (m * template_count + k) holds
    pieces of sum_s w |T|^2 for micrograph m, template k. Each subgroup sums its threads' accumulators
    (subgroup_add) and lane 0 writes one value, so the buffer and its reduction are subgroup-size times smaller;
    if the workgroup layout does not allow it (small FFTs), every thread writes its own value."""
    with vd.fft.fft_context(buffer_shape, axis=axis, compute_type=vd.complex64,
                            name=f"fused_norm_convolution_{buffer_shape}_{axis}") as ctx:
        io_manager = ctx.make_io_manager(input_map=input_map, output_map=output_map, output_type=vd.complex64,
                                         input_type=None, kernel_map=kernel_map)
        batch_threads = int(ctx.config.batch_threads)
        if ctx.config.batch_inner_count <= 1:
            raise ValueError("fused norms expect a (K, H, C) buffer with C > 1 columns")
        tid = ctx.grid.tid
        inner = int(ctx.grid.inline_batches_inner)          # adjacent columns per workgroup (local x)
        sg = int(vd.get_context().subgroup_size)
        slice_threads = inner * batch_threads                # threads of one template in a workgroup
        use_subgroups = sg > 0 and slice_threads % sg == 0 and sg % inner == 0 and columns % inner == 0
        if use_subgroups:
            col_blocks = columns // inner
            per_block = slice_threads // sg
            partials_per_row = col_blocks * per_block
        else:
            partials_per_row = columns * batch_threads

        state.acc = vc.new_float_register(0.0)

        io_manager.read_input()
        ctx.execute(inverse=False)
        ctx.register_shuffle()

        backup_registers = None
        if kernel_num > 1:
            backup_registers = ctx.allocate_registers("backup")
            backup_registers.read_from_registers(ctx.registers)

        for kern_index in range(kernel_num):
            if backup_registers is not None:
                ctx.registers.read_from_registers(backup_registers)

            set_global_kernel_index(kern_index)
            state.begin()
            io_manager.read_kernel()

            # one partial sum per thread, every thread (0 if it held no element): its FFT is (template k = outer batch,
            # column = inner batch) of the (K, H, C) buffer along axis 1. Identifying the column from a register's
            # io index instead left the slots of threads without registers unwritten (stale memory summed).
            k = (ctx.grid.global_outer_offset).to_dtype(vc.i32).to_register()
            column = (ctx.grid.global_inner_offset).to_dtype(vc.i32).to_register()
            row_base = ((kern_index * template_count + k) * partials_per_row).to_register()
            if use_subgroups:
                # all lanes of a subgroup share the template k and the column block (column // inner)
                total = vc.subgroup_add(state.acc).to_register()
                lin = (vc.local_invocation_id().x.to_dtype(vc.i32) + inner * tid.to_dtype(vc.i32)).to_register()
                with vc.if_block(vc.all(k < template_count, column < columns, lin % sg == 0)):
                    state.partials[row_base + (column // inner) * per_block + lin // sg] = total
            else:
                with vc.if_block(vc.all(k < template_count, column < columns)):
                    state.partials[row_base + column * batch_threads + tid] = state.acc

            ctx.execute(inverse=True)
            ctx.registers.normalize()
            io_manager.write_output(inverse=True)
            set_global_kernel_index(None)

    return ctx.get_callable(), partials_per_row
