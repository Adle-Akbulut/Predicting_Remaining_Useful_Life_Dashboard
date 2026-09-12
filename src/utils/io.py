import json
from pathlib import Path


def ensure_dir(path):
    """
    Create a directory if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    """
    Save a Python object as JSON.
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    """
    Load JSON from a file.
    """
    with open(path, "r") as f:
        return json.load(f) 
