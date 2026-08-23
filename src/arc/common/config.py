"""Configuration utilities for ARC."""


def validate_config(config):
    """Validate the small configuration contract used by the runners."""
    if not isinstance(config, dict):
        raise ValueError("Config must contain a mapping")
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("Config must contain a model mapping")
    block_size = model.get("block_size", 4)
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
        raise ValueError("model.block_size must be a positive integer")
    model["block_size"] = block_size
    return config


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
        except ImportError:
            import json
            return validate_config(json.loads(text))
        return validate_config(yaml.safe_load(text))
    else:
        import json
        return validate_config(json.loads(text))
