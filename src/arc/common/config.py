"""Configuration utilities for ARC."""

def load_config(path):
    """Load ARC config from path."""
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open("r") as f:
        return json.load(f)
