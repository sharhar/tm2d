import inspect
import vkdispatch as vd
import numpy as np
import tqdm
from typing import Optional

from .ctf.ctf import CTFParams

from .base_template import Template
from .base_comparator import Comparator
from .base_results import Results
from .param_set import ParamSet

class Plan:
    """Exhaustive search of templates against micrographs.

    Each submission of the command graph processes template_batch_size "slots"; per-pixel results are updated once
    per submission, so a larger template_batch_size means less result traffic per template (and more GPU memory:
    two (micrographs * template_batch_size, H, W) correlation buffers). Two ways to fill the slots:

      "per_slot": every slot holds any (rotation, CTF) combination of one pixel size (the parameter list is
          flattened), one projection per slot; full slots even with a single CTF value.
      "shared":   the slots of an instance share one rotation and hold K CTF values; one projection (or density
          sample) per instance, so it is cheaper when there are many CTF values.
      "auto" (default): "shared" if the number of CTF values is a multiple of template_batch_size, else "per_slot".
          The graph for the mode is built on first use (run) and rebuilt only if the mode changes."""
    template: Template
    comparator: Comparator
    results: Results
    rotation: Optional[np.ndarray]
    pixel_size: Optional[float]
    ctf_params: CTFParams
    template_batch_size: int
    enable_rotation_weights: bool

    _status_bar: Optional[tqdm.tqdm]

    def __init__(self,
                 template: Template,
                 comparator: Comparator,
                 results: Results,
                 rotation: Optional[np.ndarray] = None,
                 pixel_size: Optional[float] = None,
                 ctf_params: Optional[CTFParams] = None,
                 template_batch_size: int = 8,
                 output_radius: Optional[int] = None,
                 enable_rotation_weights: bool = False,
                 rotation_sharing: str = "auto"):

        if rotation_sharing not in ("auto", "per_slot", "shared"):
            raise ValueError(f"rotation_sharing must be 'auto', 'per_slot' or 'shared', not {rotation_sharing!r}")
        self.template_batch_size = int(template_batch_size)

        if ctf_params is None:
            ctf_params = CTFParams()

        self.template = template
        self.comparator = comparator
        self.results = results
        self.results.set_template_shape(self.template.get_shape())
        self.rotation = rotation
        self.pixel_size = pixel_size
        self.ctf_params = ctf_params
        self.enable_rotation_weights = enable_rotation_weights
        self.output_radius = output_radius
        self.rotation_sharing = rotation_sharing

        self._status_bar = None
        self.mode = None
        self.cmd_graph = None
        if rotation_sharing != "auto":
            self._build(rotation_sharing)

    def _build(self, mode: str):
        """Record the command graph for slot mode `mode` ("per_slot" or "shared")."""
        self.mode = None
        self.cmd_graph = vd.CommandGraph()

        prev_graph = vd.set_global_graph(self.cmd_graph)
        try:          # restore the previous global graph even if recording fails (e.g. a buffer is too large)
            self._record(mode)
        finally:
            vd.set_global_graph(prev_graph)
        self.mode = mode

    def _record(self, mode: str):

        K = self.template_batch_size
        if self.rotation is not None:
            rotations = self.template.get_rotation_matrices(self.rotation)
        elif mode == "per_slot":
            rotations = [self.cmd_graph.bind_var(f"rotation_matrix{k}") for k in range(K)]   # one per slot
        else:
            rotations = self.cmd_graph.bind_var("rotation_matrix")                            # shared by the slots

        self.template_buffer = self.template.make_template(
            rotations,
            self.cmd_graph.bind_var("pixel_size") if self.pixel_size is None else self.pixel_size,
            template_count=K,
            cmd_graph=self.cmd_graph,
            ctf_params=self.ctf_params
        )

        # with edge positions excluded, the comparator need not compute the rows / columns that hold none
        mic_buffer = getattr(self.comparator, "micrographs_buffer", None)
        trim = (mic_buffer is not None and hasattr(self.results, "edges_excluded") and self.output_radius is None
                and self.results.edges_excluded(tuple(int(n) for n in mic_buffer.real_shape[1:3])))
        # comparators without trim support (no trim_to_valid parameter) are called as before
        if "trim_to_valid" in inspect.signature(self.comparator.compare_template).parameters:
            self.comparison_buffer = self.comparator.compare_template(
                self.template_buffer, output_radius=self.output_radius, trim_to_valid=trim)
        else:
            self.comparison_buffer = self.comparator.compare_template(
                self.template_buffer, output_radius=self.output_radius)

        # factors the comparator leaves to the results stage (fused template norms), if any
        self.results.set_score_norms(getattr(self.comparator, "score_norms", None),
                                     getattr(self.comparator, "score_norm_inv_n", 1.0))

        if not self.enable_rotation_weights:
            weights = 1
        elif mode == "per_slot":
            weights = [self.cmd_graph.bind_var(f"rotation_weight{k}") for k in range(K)]
        else:
            weights = [self.cmd_graph.bind_var("rotation_weight")] * K
        self.results.check_comparison(
            self.comparison_buffer,
            weights,
            *[self.cmd_graph.bind_var(f"index{k}") for k in range(K)]
        )


    def set_data(self, data: np.ndarray, weights=None):
        """weights: optional whitening weights for the comparator (see ComparatorCrossCorrelation.set_data)."""
        self.comparator.set_data(data, weights=weights)

    def reset(self):
        self.results.reset()

    def prepare(self, params: ParamSet):
        """Build (compile) the command graph for the slot mode `params` needs, if not built yet. run() calls it; call
        it beforehand to keep graph compilation out of a timed run."""
        mode = self.choose_mode(params)
        if mode != self.mode:
            self._build(mode)

    def choose_mode(self, params: ParamSet) -> str:
        if self.rotation_sharing != "auto":
            return self.rotation_sharing
        C, K = params.get_ctf_count(), self.template_batch_size
        return "shared" if C >= K and C % K == 0 else "per_slot"

    def _submit_slots(self, params: ParamSet, pixel_index: int, p: np.ndarray, r: np.ndarray, c: np.ndarray,
                      rotation_matrices: Optional[np.ndarray]):
        """One submission: parameter indices p (rotation r, CTF c, all at pixel size pixel_index, -1 = empty slot),
        a multiple of template_batch_size long, laid into instances of template_batch_size slots. In "shared" mode
        the slots of an instance all have the same rotation."""
        K = self.template_batch_size
        n_inst = len(p) // K
        p, r, c = p.reshape(n_inst, K), r.reshape(n_inst, K), c.reshape(n_inst, K)
        shared = self.mode == "shared"

        if rotation_matrices is not None and shared:
            self.cmd_graph.set_var("rotation_matrix", rotation_matrices[r[:, 0]])
        if self.enable_rotation_weights and shared:
            self.cmd_graph.set_var("rotation_weight", params.rotations_weights[r[:, 0]].astype(np.float32))
        for k in range(K):
            if rotation_matrices is not None and not shared:
                self.cmd_graph.set_var(f"rotation_matrix{k}", rotation_matrices[r[:, k]])
            for ii, field in enumerate(params.ctf_set.field_names):
                self.cmd_graph.set_var(f"{field}_{k}",
                                       params.ctf_set.combinations_array[c[:, k], ii].astype(np.float32))
            if self.enable_rotation_weights and not shared:
                weights = params.rotations_weights[r[:, k]].astype(np.float32)
                weights[p[:, k] < 0] = 0.0
                self.cmd_graph.set_var(f"rotation_weight{k}", weights)
            self.cmd_graph.set_var(f"index{k}", p[:, k].astype(np.int32))

        if params.pixel_sizes is not None and self.pixel_size is None:
            self.cmd_graph.set_var("pixel_size", np.full(n_inst, params.pixel_sizes[pixel_index], np.float32))

        self.cmd_graph.submit_any(n_inst)

        if self._status_bar is not None:
            self._status_bar.update(int(np.sum(p >= 0)))

    def run(self,
            params: ParamSet,
            enable_progress_bar: bool = False,
            batch_size: int = 32,
            compile_results: bool = True):
        """batch_size: graph instances per submission (each instance = template_batch_size slots)."""

        if params.rotations_weights is None and self.enable_rotation_weights:
            raise ValueError("enable_rotation_weights is True, but no rotation weights provided in params.")

        self.prepare(params)

        K = self.template_batch_size
        R, S, C = params.get_rotation_count(), params.get_pixel_size_count(), params.get_ctf_count()
        rotation_matrices = None
        if params.rotations is not None and self.rotation is None:
            rotation_matrices = self.template.get_rotation_matrices(np.asarray(params.rotations, dtype=np.float32))

        if enable_progress_bar:
            self._status_bar = tqdm.tqdm(total=params.get_total_count(), dynamic_ncols=True)

        # parameter index p = (rotation * S + pixel) * C + ctf (ParamSet.get_values_at_index). A submission never
        # mixes pixel sizes (one pixel size per graph instance).
        if self.mode == "per_slot":
            rot = np.repeat(np.arange(R), C)          # slots filled freely from the flattened list
            ctf = np.tile(np.arange(C), R)
        else:
            Cp = -(-C // K) * K                       # each rotation's CTF list padded to a multiple of K
            rot = np.repeat(np.arange(R), Cp)
            ctf = np.tile(np.arange(Cp), R)
        per_submit = K * int(batch_size)
        for s in range(S):
            p_all = np.where(ctf < C, (rot * S + s) * C + ctf, -1)
            for start in range(0, len(p_all), per_submit):
                sl = slice(start, start + per_submit)
                p, r, c = p_all[sl], rot[sl], np.minimum(ctf[sl], C - 1)
                pad = (-len(p)) % K
                if pad:
                    p = np.concatenate([p, -np.ones(pad, np.int64)])
                    r = np.concatenate([r, np.zeros(pad, np.int64)])
                    c = np.concatenate([c, np.zeros(pad, np.int64)])
                self._submit_slots(params, s, p, r, c, rotation_matrices)

        if compile_results:
            # with rotation weights the per-pixel moments are weighted sums: normalize by the total weight
            total_weight = None
            if self.enable_rotation_weights:
                total_weight = float(np.sum(params.rotations_weights)) * params.get_ctf_count() * params.get_pixel_size_count()
            self.results.compile_results(params.get_total_count(), total_weight=total_weight)

        if self._status_bar is not None:
            self._status_bar.close()
            self._status_bar = None
