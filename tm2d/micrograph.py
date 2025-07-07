import numpy as np
import tm2d

import vkdispatch as vd

from typing import Tuple

import tm2d.utilities as tu

layout_to_number = {
    (False, False): 0,
    (False, True): 1,
    (True, False): 2,
    (True, True): 3
}

class StaticSignal2D:
    data: np.ndarray
    backing_buffer: vd.Buffer
    initial_layout: int
    needed_layout: int
    current_layout: int
    shape: Tuple[int, int]

    def __init__(self, width: int, height: int, fft: bool, shifted: bool) -> None:
        self.backing_buffer = vd.Buffer((width, height), vd.complex64)
        self.shape = (width, height)
        self.initial_layout = layout_to_number[(fft, shifted)]
        self.needed_layout = None
        self.data = None

    def buffer(self):
        return self.backing_buffer

    def set_data(self, data: np.ndarray) -> None:
        self.data = data

        if self.needed_layout is not None:
            self.do_formating()
    
    def layout_number_increment(self) -> None:
        if self.current_layout == 3:
            raise ValueError("Cannot increment layout number, already at maximum")
        
        if self.current_layout == 0:
            self.data = np.fft.fftshift(self.data)
        elif self.current_layout == 1:
            self.data = np.fft.fft2(self.data)
        elif self.current_layout == 2:
            self.data = np.fft.fftshift(self.data)

        self.current_layout += 1
    
    def layout_number_decrement(self) -> None:
        if self.current_layout == 0:
            raise ValueError("Cannot increment layout number, already at maximum")
        
        if self.current_layout == 3:
            self.data = np.fft.ifftshift(self.data)
        elif self.current_layout == 2:
            self.data = np.fft.ifft2(self.data)
        elif self.current_layout == 1:
            self.data = np.fft.ifftshift(self.data)

        self.current_layout -= 1

    def do_formating(self):
        self.current_layout = self.initial_layout

        if self.current_layout < self.needed_layout:
            while self.current_layout < self.needed_layout:
                self.layout_number_increment()
        elif self.current_layout > self.needed_layout:
            while self.current_layout > self.needed_layout:
                self.layout_number_decrement()
        
        self.backing_buffer.write(self.data)

    def require_layout(self, fft: bool, shifted: bool) -> None:
        if self.needed_layout is not None:
            raise ValueError("Data already formated")

        self.needed_layout = layout_to_number[(fft, shifted)]

        if self.data is not None:
            self.do_formating()

def make_micrograph_signal_2d(
        data: np.ndarray, 
        whiten: bool = True,
        normalize_input: bool = True, 
        remove_outliers: bool = True, 
        n_std = 5) -> StaticSignal2D:
    assert data.ndim == 2

    if normalize_input:
        reference_image = data.copy()
        test_image_normalized: np.ndarray = tu.normalize_image(reference_image, n_std=n_std, remove_outliers=remove_outliers)
    else:
        test_image_normalized: np.ndarray = data.copy() # don't actually normalize

    if whiten:
        test_image_normalized = tu.whiten_image(test_image_normalized)

    signal = StaticSignal2D(test_image_normalized.shape[0], test_image_normalized.shape[1])
    signal.set_data(test_image_normalized, False, False)

    return signal
    