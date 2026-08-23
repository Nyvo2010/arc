"""Configuration utilities for ARC."""

def load_config(path):
    """Load ARC config from path. Supports JSON and YAML."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
            return yaml.safe_load(text)
        except Exception:
            # fallback to json
            import json
            return json.loads(text)
    else:
        import json
        return json.loads(text)
