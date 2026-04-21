from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapTokens:
    """Memory-mapped token stream stored as uint32 in a .bin file.

    This is designed to be safe with DataLoader workers: the memmap is re-opened
    lazily after pickling.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._mm: np.memmap | None = None

    def _ensure_open(self) -> None:
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint32, mode="r")
            if self._mm.size < 1:
                raise ValueError(f"{self.path} is empty.")

    def __len__(self) -> int:
        self._ensure_open()
        return int(self._mm.size)  # type: ignore[union-attr]

    def slice_u32(self, start: int, length: int) -> np.ndarray:
        self._ensure_open()
        return self._mm[start : start + length]  # type: ignore[union-attr]

    def __getstate__(self):
        return {"path": self.path}

    def __setstate__(self, state):
        self.path = state["path"]
        self._mm = None


@dataclass
class DeterministicMixedPretokBatches(Dataset):
    """
    Deterministic by global_step: __getitem__(idx) corresponds to global_step = idx + step_offset.
    Each item returns (x, y) where x,y are shaped (B,T).
    """

    sources: Dict[str, MemmapTokens]
    weights: Dict[str, float]
    seq_len: int
    batch_size: int
    seed: int = 1337
    step_offset: int = 0
    rank: int = 0
    world_size: int = 1

    def __post_init__(self):
        self.names = list(self.sources.keys())
        w = torch.tensor([self.weights[n] for n in self.names], dtype=torch.float32)
        self.w = (w / w.sum()).cpu()

        need = int(self.seq_len) + 1
        self.max_starts = []
        for n in self.names:
            mx = len(self.sources[n]) - need
            if mx <= 0:
                raise ValueError(f"Not enough tokens in {n}: need > {need}, have {len(self.sources[n])}")
            self.max_starts.append(mx)

        self._length = 10**12  # effectively infinite

    def set_step_offset(self, step_offset: int) -> None:
        self.step_offset = int(step_offset)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        global_step = int(idx) + int(self.step_offset)

        g = torch.Generator()
        # rank-aware deterministic seed (matches Phase 2 notebook)
        g.manual_seed(int(self.seed) + global_step * int(self.world_size) + int(self.rank))

        # which source each sample comes from
        choices = torch.multinomial(self.w, int(self.batch_size), replacement=True, generator=g)

        x = torch.empty((int(self.batch_size), int(self.seq_len)), dtype=torch.long)
        y = torch.empty((int(self.batch_size), int(self.seq_len)), dtype=torch.long)

        for i, c in enumerate(choices.tolist()):
            name = self.names[c]
            max_start = self.max_starts[c]

            start = int(torch.randint(0, max_start, (1,), generator=g).item())
            chunk_u32 = self.sources[name].slice_u32(start, int(self.seq_len) + 1)

            # torch.from_numpy prefers a writable array; memmap slices are often read-only.
            arr = np.asarray(chunk_u32)
            if not arr.flags.writeable:
                arr = arr.copy()
            chunk = torch.from_numpy(arr).to(torch.int64)
            x[i] = chunk[:-1]
            y[i] = chunk[1:]

        return x, y

