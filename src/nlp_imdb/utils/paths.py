# src/nlp_imdb/utils/paths.py

from pathlib import Path


class ProjectPaths:
    """
    Centralized project path management.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.cwd()

        self.configs = self.root / "configs"
        self.data = self.root / "data"
        self.data_raw = self.data / "raw"
        self.data_processed = self.data / "processed"
        self.models = self.root / "models"
        self.reports = self.root / "reports"

    def ensure_dirs(self) -> None:
        """
        Create directories if they do not exist.
        """
        for path in [
            self.data_raw,
            self.data_processed,
            self.models,
            self.reports,
        ]:
            path.mkdir(parents=True, exist_ok=True)
