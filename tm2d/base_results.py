import numpy as np
import vkdispatch as vd
import vkdispatch.codegen as vc

def valid_center_range(n: int, m: int) -> tuple:
    """Inclusive range of map indices (results maps, centered convention) at which a template of size m lies fully
    inside an image of size n. A template with top-left corner at t peaks at index t + m // 2 (the micrograph pixel
    under the template's center, for even and odd n), and t runs over 0 .. n - m (a "valid" correlation: the template lies fully inside the image).
    Results maps are the raw circular correlation fftshifted: map index i = (raw index + n // 2) % n."""
    lo = m // 2
    return lo, lo + n - m

class Results:
    def __init__(self):
        raise NotImplementedError("Results is an abstract class. Please implement it in a subclass.")

    def check_comparison(self, comparison_buffer: vd.RFFTBuffer, rotation_weights: vc.Var[vc.f32], *indices: vc.Var[vc.i32]):
        raise NotImplementedError("check_comparison must be implemented in a subclass.")

    def reset(self):
        """
        Reset the results to their initial state.
        This method should be implemented in subclasses to reset any internal state.
        """
        raise NotImplementedError("reset must be implemented in a subclass.")

    template_shape = None
    output_radius = None
    exclude_edges_setting = "auto"
    score_norms = None
    score_norm_inv_n = 1.0

    def set_score_norms(self, score_norms, inv_n: float = 1.0):
        """Called by Plan before check_comparison: a (micrographs * templates,) float buffer of sum_s w |T|^2 that the
        comparator left to the results stage (fused norms): each score is multiplied by 1 / sqrt(norm * inv_n).
        None if the maps are already scaled."""
        self.score_norms = score_norms
        self.score_norm_inv_n = float(inv_n)

    def set_template_shape(self, template_shape: tuple):
        """Called by Plan before check_comparison: the (h, w) template box, used to mark map positions where the
        template crosses the image edge (and so wraps around onto the opposite edge) as invalid."""
        self.template_shape = (int(template_shape[0]), int(template_shape[1]))

    @staticmethod
    def _check_edge_setting(exclude_edges):
        if exclude_edges == "auto" or isinstance(exclude_edges, (bool, np.bool_)):
            return exclude_edges if exclude_edges == "auto" else bool(exclude_edges)
        raise ValueError(f"exclude_edges must be True, False or 'auto', not {exclude_edges!r}")

    def edges_excluded(self, image_shape: tuple) -> bool:
        """Whether positions where the template crosses the image edge are excluded, for an (H, W) image.
        exclude_edges=True / False as given; "auto" (default): excluded for micrograph searches (template smaller than
        the image in both dimensions, no output_radius; valid positions only), not for particle boxes (template as
        large as the image, or a local search with output_radius: every position is scored, the circular
        correlation, as in tm2d v0.3.0)."""
        if self.exclude_edges_setting is True and self.template_shape is None:
            raise ValueError("exclude_edges=True needs the template shape: call set_template_shape() (Plan does)")
        if self.exclude_edges_setting != "auto":
            return bool(self.exclude_edges_setting)
        if self.template_shape is None or self.output_radius is not None:
            return False
        return self.template_shape[0] < int(image_shape[0]) and self.template_shape[1] < int(image_shape[1])

    def compile_results(self, templates_count: int, total_weight: float = None):
        """
        Compile the results after all comparisons have been made.
        This method should be implemented in subclasses to compile the results into a final form that can be easily accessed.
        """
        raise NotImplementedError("compile_results must be implemented in a subclass.")