import os
from dataclasses import dataclass, field
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("CONTROLPLANE_DATA_DIR", PLATFORM_DIR / "var")))
    retry_limit: int = field(default_factory=lambda: int(os.environ.get("CONTROLPLANE_RETRY_LIMIT", "3")))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "controlplane.db"

    def project_dir(self, pid: str) -> Path:
        return self.data_dir / "projects" / pid
