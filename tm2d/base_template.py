import vkdispatch as vd
import vkdispatch.codegen as vc
import numpy as np

from typing import Union, List, Tuple
from .ctf.ctf import CTFParams

def per_slot_rotations(rotations, template_count: int) -> list:
    """The rotation of each of the template_count slots of a template batch, as a list.

    rotations: a list with one entry (graph variable or 4x4 matrix) per slot; a single graph variable or 4x4
    matrix (all slots share it); or an (N, 4, 4) array with N == 1 (shared) or N == template_count (one each)."""
    if isinstance(rotations, (list, tuple)):
        if len(rotations) != template_count:
            raise ValueError(f"{len(rotations)} rotations given for {template_count} template slots")
        return list(rotations)
    if isinstance(rotations, np.ndarray):
        if rotations.ndim == 2:
            return [rotations] * template_count
        if rotations.shape[0] == 1:
            return [rotations[0]] * template_count
        if rotations.shape[0] == template_count:
            return [rotations[k] for k in range(template_count)]
        raise ValueError(f"{rotations.shape[0]} rotation matrices given for {template_count} template slots")
    return [rotations] * template_count

class Template:
    """Base class of tm2d templates. A subclass implements _make_template, _get_rotation_matrices and get_shape.

    supports_ctf_free: True if _make_template takes apply_ctf=False and then renders projections without the CTF
    (the image-side CTF and prefiltered searches need them). A subclass that sets it must honour the flag; Plan
    never uses the image side for a template without it."""
    supports_ctf_free: bool = False

    def __init__(self):
        raise NotImplementedError("Template is an abstract class. Please implement it in a subclass.")

    def _make_template(self,
                      rotations: Union[vc.Var[vc.m4], np.ndarray],
                      pixel_size: float,
                      ctf_params: CTFParams,
                      template_count: int = 1,
                      cmd_graph: vd.CommandGraph = None,
                      apply_ctf: bool = True) -> vd.RFFTBuffer:
        """
        This abstract method should return a buffer containing the sampled and filtered template in real space.
        """

        raise NotImplementedError("make_template must be implemented in a subclass.")

    def make_template(self,
                      rotations: Union[vc.Var[vc.m4], List[int], np.ndarray],
                      pixel_size: float,
                      ctf_params: CTFParams = None,
                      template_count: int = 1,
                      cmd_graph: vd.CommandGraph = None,
                      apply_ctf: bool = True) -> vd.RFFTBuffer:
        """Record the rendering of template_count templates (real space, (template_count, h, w) RFFTBuffer).

        apply_ctf=False: projections without the CTF (other filters, e.g. the atomic template's Gaussian, still
        apply). Only for templates with supports_ctf_free; the template's own disable_ctf setting is not changed.
        """
        if not apply_ctf and not self.supports_ctf_free:
            raise NotImplementedError(f"{type(self).__name__} cannot render CTF-free projections "
                                      "(supports_ctf_free is False)")

        if ctf_params is None:
            ctf_params = CTFParams()

        # a list of angles / matrices becomes an array; a list of graph variables (one per slot) passes through
        if isinstance(rotations, list) and all(isinstance(r, (int, float, np.number, list, tuple, np.ndarray)) for r in rotations):
            rotations = np.array(rotations, dtype=np.float32)

        if isinstance(rotations, np.ndarray):
            if rotations.ndim == 1:
                rotations = rotations.reshape(1, 3)

            if rotations.ndim == 2:
                if rotations.shape[1] == 4 and rotations.shape[0] == 4:
                    rotations = rotations.reshape(1, 4, 4)
                else:
                    assert rotations.shape[1] == 3 , "Rotations must be a 2D array with shape (N, 3)."

                    rotations = self.get_rotation_matrices(rotations)

        return self._make_template(
            rotations=rotations,
            pixel_size=pixel_size,
            ctf_params=ctf_params,
            template_count=template_count,
            cmd_graph=cmd_graph,
            **({} if apply_ctf else {"apply_ctf": False})   # subclasses without the flag keep working
        )

    def _get_rotation_matrices(self, rotations: np.ndarray) -> np.ndarray:
        raise NotImplementedError("get_rotation_matrices must be implemented in a subclass.")

    def get_rotation_matrices(self, rotations: Union[np.ndarray, Tuple[float, float, float]]) -> np.ndarray:
        if not isinstance(rotations, np.ndarray):
            rotations = np.array(rotations, dtype=np.float32).reshape(1, 3)

        if rotations.ndim == 1:
            rotations = rotations.reshape(1, 3)

        return self._get_rotation_matrices(rotations)

    def get_shape(self) -> tuple:
        """
        Return the shape of the template.
        This method should be implemented in subclasses to return the correct shape.
        """
        raise NotImplementedError("get_shape must be implemented in a subclass.")