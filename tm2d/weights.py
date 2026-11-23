import dataclasses

import numpy as np


@dataclasses.dataclass
class RadialWeights:
    """Radially symmetric whitening weights w(s), s = |spatial frequency| in cycles per pixel.

    values: w sampled at s_j = j * ds, j = 0 .. n - 1; shape (n,) for all micrographs, or (M, n) one table per
    micrograph. Between samples w is linear in s; beyond the last sample it keeps the last value. The comparator
    evaluates w the same way on the CPU (to weight the image) and on the GPU (for the template norms), so the two
    always agree. tm2d_utils.whitening_radial_weights builds one from a micrograph's noise spectrum (w = 1 / P)."""
    values: np.ndarray
    ds: float

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.values.ndim not in (1, 2) or self.values.shape[-1] < 2:
            raise ValueError("values must have shape (n,) or (M, n) with n >= 2")
        if not np.all(np.isfinite(self.values)) or np.any(self.values < 0):
            raise ValueError("weights must be finite and non-negative")
        self.ds = float(self.ds)
        if not self.ds > 0:
            raise ValueError("ds must be positive")

    @classmethod
    def ones(cls) -> "RadialWeights":
        return cls(np.ones(2), 1.0)

    def tables(self, count: int) -> np.ndarray:
        """(count, n) table, one row per micrograph."""
        if self.values.ndim == 1:
            return np.broadcast_to(self.values, (count, self.values.shape[0])).copy()
        if self.values.shape[0] != count:
            raise ValueError(f"{self.values.shape[0]} weight tables for {count} micrographs")
        return self.values.copy()

    def evaluate(self, shape: tuple, count: int = None) -> np.ndarray:
        """w on the rfft2 half-plane of an (H, W) image, unshifted, with DC and the Nyquist lines set to 0.
        Shape (H, W//2+1), or (count, H, W//2+1) if count is given."""
        H, W = shape
        sy = np.fft.fftfreq(H)[:, None]
        sx = np.fft.rfftfreq(W)[None, :]
        j = np.minimum(np.sqrt(sy * sy + sx * sx) / self.ds, self.values.shape[-1] - 1)
        j0 = np.minimum(np.floor(j).astype(np.int64), self.values.shape[-1] - 2)
        f = j - j0
        mask = edge_mask(shape)
        tabs = self.tables(1 if count is None else count)
        out = tabs[:, j0] * (1 - f) + tabs[:, j0 + 1] * f
        out[:, mask] = 0.0
        return out[0] if count is None else out


def edge_mask(shape):
    """True at DC and on the Nyquist row and column (even sizes) of the rfft2 half-plane of an (H, W) image."""
    H, W = shape
    m = np.zeros((H, W // 2 + 1), bool)
    m[0, 0] = True
    if H % 2 == 0:
        m[H // 2, :] = True
    if W % 2 == 0:
        m[:, -1] = True
    return m
