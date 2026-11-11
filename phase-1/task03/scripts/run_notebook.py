"""Execute a notebook top to bottom and save the outputs in place."""
import sys
from pathlib import Path

# pyrefly: ignore[missing-import]
import nbformat

path = Path(sys.argv[1]).resolve()
nb = nbformat.read(path, as_version=4)

try:
    # pyrefly: ignore[missing-import]
    from nbconvert.preprocessors import ExecutePreprocessor
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(path.parent)}}
    )
except ImportError:
    # pyrefly: ignore[missing-import]
    from nbclient import NotebookClient
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()

nbformat.write(nb, path)
print("executed", path.name)

