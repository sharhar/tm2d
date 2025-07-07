import numpy as np
import vkdispatch as vd 
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

import tm2d
import tm2d.utilities as tu

@vd.shader(exec_size=lambda args: args.buffer.size)
def fill_buffer(buffer: Buff[c64], value: Const[c64]):
    tid = vc.global_invocation().x
    buffer[tid] = value

def make_ctf_engine(
    input_shape,
    pix_size,
    ctf_params,
    ):

    # initialize command stream
    cmd_stream = vd.CommandStream()
    prev_stream = vd.set_global_cmd_stream(cmd_stream)

    # make initial buffer
    ctf_output = tm2d.Signal2D(input_shape[0], input_shape[1], True, False)
    fill_buffer(ctf_output.buffer(), 0.5)

    # apply ctf
    tm2d.apply_ctf(
        ctf_output,
        pix_size,
        cmd_stream.bind_var('defocus'),
        ctf_params,
    )

    # require shifted and fourier space
    ctf_output.require_layout(True, True)

    # finish command stream
    vd.set_global_cmd_stream(prev_stream)

    return ctf_output, cmd_stream

def calc_ctf(
    defocus: float,
    ctf_output: vd.Buffer,
    cmd_stream: vd.CommandStream,
    ):

    cmd_stream.set_var('defocus', np.array([defocus, 0, 0, 0], dtype=np.float32))
    cmd_stream.submit()

    return ctf_output.buffer().read(0).real