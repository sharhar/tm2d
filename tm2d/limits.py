"""Device size limits that vkdispatch does not check itself.

A storage buffer larger than the device's maxStorageBufferRange (4 GiB - 1 on NVIDIA's Vulkan driver) is allocated
without error, but shaders silently read / write garbage beyond the limit (found 2026-11-22: 8 x 4096^2 micrographs
with 8 templates per batch gave MIP 0 on 6 of 8 micrographs; a 1024^3 density volume gave no scores at all). tm2d
checks its large allocations against the limit and raises instead.
"""
import numpy as np
import vkdispatch as vd


def max_buffer_bytes() -> int:
    """Smallest maxStorageBufferRange over the context's devices (bytes)."""
    return int(min(d.max_storage_buffer_range for d in vd.get_context().device_infos))


def check_buffer_bytes(shape, bytes_per_element: int, what: str, hint: str = "") -> None:
    """Raise ValueError if a buffer of this shape and element size exceeds the device limit."""
    nbytes = int(np.prod([int(s) for s in shape], dtype=np.int64)) * int(bytes_per_element)
    limit = max_buffer_bytes()
    if nbytes > limit:
        raise ValueError(f"{what} {tuple(int(s) for s in shape)} needs {nbytes / 2 ** 30:.2f} GiB, more than the device's "
                         f"largest storage buffer ({limit / 2 ** 30:.2f} GiB); shaders would silently compute garbage. {hint}")
