import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

class World:
    def __init__(self, path: str, max_dist: float, eps: float = 1e-6):
        self.lines: NDArray[np.float64]
        self.max_dist: float
        self.eps: float
    @classmethod
    def from_lines(
        cls, lines: NDArray[np.float64], max_dist: float, eps: float = 1e-6
    ) -> "World": ...
    def skymask(self, pos: Tuple[float, float]) -> SkymaskMap: ...
    def par_samples(
        self, pos: List[Tuple[float, float]], samples: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

class SkymaskMap:
    def at(self, theta: float) -> float: ...
    def samples(self, samples: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def segments(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]: ...
