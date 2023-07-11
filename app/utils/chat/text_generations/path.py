from pathlib import Path
from typing import Optional

from app.common.config import BASE_DIR


def resolve_model_path_to_posix(
    model_path: str, default_relative_directory: Optional[str] = None
):
    """Resolve a model path to a POSIX path, relative to the BASE_DIR."""
    path = Path(model_path)
    parent_directory: Path = (
        Path(BASE_DIR) / Path(default_relative_directory)
        if default_relative_directory is not None
        else Path(BASE_DIR)
        if Path.cwd() == path.parent.resolve()
        else path.parent.resolve()
    )
    filename: str = path.name
    return (parent_directory / filename).as_posix()
