import json
import numpy as np
from pathlib import Path

class GridMap:
    def __init__(self, layout: np.ndarray, dx: float):
        self.grid = layout.astype(bool).copy()
        self.dx = dx
        
    @classmethod
    def from_file(cls, filepath: Path | str):
        with np.load(filepath) as data:
            grid = data["grid"]
            metadata = json.loads(data["metadata"].item())
        return cls(layout=grid, dx=metadata["dx"])

    def write(self, fpath: Path | str):
        metadata = json.dumps({
            "dx": self.dx
        })
        np.savez_compressed(fpath, grid=self.grid, metadata=metadata)