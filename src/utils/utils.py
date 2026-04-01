from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_config(path: Path = PROJECT_ROOT / "config.yml") -> dict:
    """Load YAML configuration file."""
    with path.open() as f:
        return yaml.safe_load(f)