"""Lightweight logger: always writes JSONL to disk; optionally forwards to wandb."""
import os
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Logger:
    """Writes metrics to a JSONL file and optionally to wandb.

    Each call to ``log`` appends one JSON record (with a ``step`` field) to
    ``<log_dir>/metrics.jsonl``.  This format is trivially readable by pandas:

        df = pd.read_json("metrics.jsonl", lines=True)
    """

    def __init__(
        self,
        log_dir: str,
        use_wandb: bool = False,
        project: str = None,
        run_name: str = None,
        config: dict = None,
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        if use_wandb and not WANDB_AVAILABLE:
            print("[Logger] wandb not installed — falling back to file-only logging.")

        self._file = open(os.path.join(log_dir, "metrics.jsonl"), "a")

        if self.use_wandb:
            wandb.init(
                project=project,
                name=run_name,
                config=config,
                dir=log_dir,
                reinit=True,
            )

    def log(self, metrics: dict, step: int = None):
        record = {"step": step, **metrics}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def save_config(self, config: dict):
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def close(self):
        self._file.close()
        if self.use_wandb:
            wandb.finish()
