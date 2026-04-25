import vkdispatch as vd
import vkdispatch.codegen as vc

class Results:
    def __init__(self):
        raise NotImplementedError("Results is an abstract class. Please implement it in a subclass.")

    def check_comparison(self, comparison_buffer: vd.RFFTBuffer, *indicies: vc.Var[vc.i32]):
        raise NotImplementedError("check_comparison must be implemented in a subclass.")

    def reset(self):
        """
        Reset the results to their initial state.
        This method should be implemented in subclasses to reset any internal state.
        """
        raise NotImplementedError("reset must be implemented in a subclass.")

    def compile_results(self, templates_count: int):
        """
        Compile the results after all comparisons have been made.
        This method should be implemented in subclasses to compile the results into a final form that can be easily accessed.
        """
        raise NotImplementedError("compile_results must be implemented in a subclass.")