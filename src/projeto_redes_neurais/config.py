from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ProjectPaths:
    root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = root / "data"
    raw_dir: Path = data_dir / "raw"
    interim_dir: Path = data_dir / "interim"
    processed_dir: Path = data_dir / "processed"
    artifacts_dir: Path = root / "artifacts"
    reports_dir: Path = root / "reports"

    def ensure(self) -> None:
        for path in (
            self.data_dir,
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.artifacts_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class CelebASpoofConfig:
    dataset_root: Path
    split: str = "train"
    image_size: int = 224
    max_samples: int | None = None
    label_key: str = "live_spoof"


DEFAULT_PATHS = ProjectPaths()
