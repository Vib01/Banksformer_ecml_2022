"""Colab-safe training entrypoint.

Minimal safe wrapper that:
- Loads a YAML config (configs/banksformer_free.yaml)
- Sets deterministic seed via train_seed.set_seed if available
- Creates required directories (checkpoints, logs)
- Calls a `train(config)` function imported from the project's training module

This file is intentionally small so it can be run from Colab or locally without
modifying upstream training code.
"""
import os
import sys
import logging
from pathlib import Path

try:
    import yaml
except Exception:
    print("PyYAML is required. Install with: pip install pyyaml")
    raise

# optional deterministic seed helper (added earlier)
try:
    from train_seed import set_seed
except Exception:
    def set_seed(seed):
        print("train_seed.set_seed not available; skipping seed setup")


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str = "configs/banksformer_free.yaml"):
    logging.basicConfig(level=logging.INFO)

    if not Path(config_path).exists():
        logging.error("Config file not found: %s", config_path)
        sys.exit(2)

    config = load_config(config_path)

    # Set deterministic seed if provided
    seed = config.get("seed", None)
    if seed is not None:
        try:
            set_seed(int(seed))
            logging.info("Set deterministic seed: %s", seed)
        except Exception as e:
            logging.warning("Failed to set seed: %s", e)

    # Create directories
    paths = config.get("paths", {})
    ckpt_dir = paths.get("checkpoints", "checkpoints")
    logs_dir = paths.get("logs", "logs")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Resume logic placeholder: later we can inspect ckpt_dir for latest checkpoint
    resume = config.get("resume", False)
    if resume:
        logging.info("Resume enabled - training will attempt to restore from checkpoints in %s", ckpt_dir)

    # Import train function from the project's training module. The real project
    # may expose a function called `train(config)`. If not present, raise a helpful error.
    try:
        from train import train
    except Exception:
        try:
            # some repos may place training in a module; try `train.train`
            import train as _train_mod
            train = getattr(_train_mod, "train", None)
        except Exception:
            train = None

    if train is None:
        logging.error(
            "Could not import a `train(config)` function. Please ensure your training entrypoint exposes `train(config)`."
        )
        # Exit non-zero so Colab/jobs fail loudly
        sys.exit(3)

    # Call the training function
    logging.info("Starting training with config: %s", config_path)
    train(config)


if __name__ == "__main__":
    cfg = os.environ.get("BANKSFORMER_CONFIG", "configs/banksformer_free.yaml")
    main(cfg)
