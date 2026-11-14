import os
from dataclasses import dataclass, field
from pathlib import Path

from .runner import Limits

PLATFORM_DIR = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"CONTROLPLANE_{name}", default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(PLATFORM_DIR / "var"))))
    retry_limit: int = field(default_factory=lambda: int(_env("RETRY_LIMIT", "3")))
    mlflow_uri: str = field(default_factory=lambda: _env("MLFLOW_URI", "http://127.0.0.1:5000"))
    agents_url: str = field(default_factory=lambda: _env("AGENTS_URL", "http://127.0.0.1:8100"))
    autostart: bool = field(default_factory=lambda: _env("AUTOSTART", "1") == "1")
    url: str = field(default_factory=lambda: _env("URL", "http://127.0.0.1:8000"))
    limits: Limits = field(default_factory=lambda: Limits(
        cpu_s=int(_env("CPU_S", "1800")), memory_mb=int(_env("MEMORY_MB", "8192")), wall_s=int(_env("WALL_S", "3600"))))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "controlplane.db"

    def project_dir(self, pid: str) -> Path:
        return self.data_dir / "projects" / pid
