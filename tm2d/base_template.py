import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np

from typing import Union, List, Tuple
from .ctf.ctf import CTFParams

class Template:
    def __init__(self):
        raise NotImplementedError("Template is an abstract class. Please implement it in a subclass.")

    def _make_template(self,
                      rotations: Union[vc.Var[vc.m4], np.ndarray],
                      pixel_size: float,
                      ctf_params: CTFParams,
                      template_count: int = 1,
                      cmd_graph: vd.CommandGraph = None,
                      disable_ctf: bool = False) -> vd.RFFTBuffer:
        """
        This abstract method should return a buffer containing the sampled and filtered template in real space.
        """

        raise NotImplementedError("make_template must be implemented in a subclass.")

    def make_template(self,
                      rotations: Union[vc.Var[vc.m4], List[int], np.ndarray],
                      pixel_size: float,
                      ctf_params: CTFParams = None,
                      template_count: int = 1,
                      cmd_graph: vd.CommandGraph = None) -> vd.RFFTBuffer:
        """
        This abstract method should return a buffer containing the sampled and filtered template in real space.
        """

        if ctf_params is None:
            ctf_params = CTFParams()

        if isinstance(rotations, list):
            rotations = np.array(rotations, dtype=np.float32)

        if isinstance(rotations, np.ndarray):
            if rotations.ndim == 1:
                rotations = rotations.reshape(1, 3)

            if rotations.ndim == 2:
                if rotations.shape[1] == 4 and rotations.shape[0] == 4:
                    rotations = rotations.reshape(1, 4, 4)
                else:
                    assert rotations.shape[1] == 3 , "Rotations must be a 2D array with shape (N, 3)."

                    rotations = self.get_rotation_matricies(rotations)

        return self._make_template(
            rotations=rotations,
            pixel_size=pixel_size,
            ctf_params=ctf_params,
            template_count=template_count,
            cmd_graph=cmd_graph
        )

    def _get_rotation_matricies(self, rotations: np.ndarray) -> np.ndarray:
        raise NotImplementedError("get_rotation_matricies must be implemented in a subclass.")

    def get_rotation_matricies(self, rotations: Union[np.ndarray, Tuple[float, float, float]]) -> np.ndarray:
        if not isinstance(rotations, np.ndarray):
            rotations = np.array(rotations, dtype=np.float32).reshape(1, 3)

        if rotations.ndim == 1:
            rotations = rotations.reshape(1, 3)

        return self._get_rotation_matricies(rotations)

    def get_shape(self) -> tuple:
        """
        Return the shape of the template.
        This method should be implemented in subclasses to return the correct shape.
        """
        raise NotImplementedError("get_shape must be implemented in a subclass.")