import vkdispatch as vd
import numpy as np

class Comparator:
    def __init__(self):
        raise NotImplementedError("Comparator is an abstract class. Please implement it in a subclass.")

    def set_data(self, data: np.ndarray):
        """
        This method should set the data buffer that will be used for comparison.
        """
        raise NotImplementedError("set_data must be implemented in a subclass.")

    def compare_template(self, template: vd.RFFTBuffer, normalize: bool = True) -> vd.RFFTBuffer:
        raise NotImplementedError("compare_template must be implemented in a subclass.")