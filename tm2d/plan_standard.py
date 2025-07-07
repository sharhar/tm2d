from .plan import Plan, Template
from .ctf import CTFParams

from .cross_correlation import ComparatorCrossCorrelation
from .results_per_pixel import ResultsPixel

class PlanStandard(Plan):
    """
    A standard plan for template matching using cross-correlation and per-pixel results.
    """

    def __init__(self,
                 template: Template,
                 data_shape: tuple,
                 pixel_size: float,
                 ctf_params: CTFParams = None,
                 template_batch_size: int = 2):
        super().__init__(
            template,
            ComparatorCrossCorrelation(data_shape, template.get_shape()),
            ResultsPixel(data_shape),
            pixel_size,
            ctf_params,
            template_batch_size
        )