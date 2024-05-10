from typing import Union, Dict
from torch import Tensor
from enum import Enum
import argparse
from omegaconf import OmegaConf
from omegaconf import DictConfig, ListConfig


class TrainerState(Enum):
    STARTING = 0
    TRAINING = 1
    VALIDATE = 2
    TERMINATE = 3


class AverageMeter:
    """Computes and stores the average and current value.

    Example:
        >>> stats = AverageMeter()
        >>> acc1 = torch.tensor(0.99) # coming from K.metrics.accuracy
        >>> stats.update(acc1, n=1)  # where n is batch size usually
        >>> round(stats.avg, 2)
        0.99
    """

    val: Union[float, bool, Tensor]
    _avg: Union[float, Tensor]
    sum: Union[float, Tensor]
    count: int

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self._avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: Union[float, bool, Tensor], n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self._avg = self.sum / self.count

    @property
    def avg(self) -> float:
        if isinstance(self._avg, Tensor):
            return float(self._avg.item())
        return self._avg


class StatsTracker:
    """
    Stats tracker for computing metrics on the fly.
    example:
        >>> stats = StatsTracker()
        >>> stats.update("acc", 0.99, 1)
        >>> stats.update("loss", 0.01, 1)
        >>> stats
        {'acc': AverageMeter1, 'loss': AverageMeter2}
        then you can access the stats by stats.stats['acc'].avg or stats.stats['acc'].val
    """

    def __init__(self) -> None:
        self._stats: Dict[str, AverageMeter] = {}

    @property
    def stats(self) -> Dict[str, AverageMeter]:
        return self._stats

    def update(self, key: str, val: float, batch_size: int) -> None:
        """Update the stats by the key value pair."""
        if key not in self._stats:
            self._stats[key] = AverageMeter()
        self._stats[key].update(val, batch_size)

    def update_from_dict(self, dic: Dict[str, float], batch_size: int) -> None:
        """Update the stats by the dict."""
        for k, v in dic.items():
            self.update(k, v, batch_size)

    def __repr__(self) -> str:
        return " ".join([
            f"{k.upper()}: {v.val:.2f} {v.val:.2f} "
            for k, v in self._stats.items()
        ])

    def as_dict(self) -> Dict[str, AverageMeter]:
        """Return the dict format."""
        return self._stats


def get_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--file',
                        type=str,
                        required=True,
                        help="Path to the YAML configuration file")
    return parser.parse_args()


def load_config(filename) -> Union[DictConfig, ListConfig]:
    config = OmegaConf.load(filename)
    return config
