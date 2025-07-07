import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

@vd.shader(exec_size=lambda args: args.output.size)
def fftshift(output: Buff[c64], input: Buff[c64]):
    ind = vc.global_invocation().x.cast_to(vd.int32).copy()

    out_x = (ind / output.shape.y).copy()
    out_y = (ind % output.shape.y).copy()

    in_x = ((out_x + input.shape.x / 2) % output.shape.x).copy()
    in_y = ((out_y + input.shape.y / 2) % output.shape.y).copy()

    output[ind] = input[in_x, in_y]