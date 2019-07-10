from bz2 import open as bz2_open
from pathlib import Path
from pickle import load as pickle_load
from typing import Any, Dict, List


from formatml.data.instance import Instance
from formatml.datasets.dataset import Dataset


class CodRepDataset(Dataset):
    def __init__(self, *, input_dir: Path, instance: Instance) -> None:
        self.instance = instance
        self._pickles = sorted(input_dir.rglob("*.pickle.bz2"), key=str)

    def collate(self, tensors: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.instance.collate(tensors)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        with bz2_open(self._pickles[index], "rb") as fh:
            return pickle_load(fh)

    def __len__(self) -> int:
        return len(self._pickles)
