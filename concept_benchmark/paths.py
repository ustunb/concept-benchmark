"""Directory paths."""
from pathlib import Path

repo_dir = Path(__file__).absolute().parent.parent
DATA_DIR = repo_dir / 'data/'
DATA_DIR .mkdir(exist_ok=True)