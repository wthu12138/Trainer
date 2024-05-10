from math import inf
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import os
from torch.nn import Module
from trainer_utils import TrainerState, AverageMeter
from accelerate import Accelerator


class EarlyStopping:
    """Callback that evaluates whether there is improvement in the loss function.

    The module track the losses and in case of finish patience sends a termination signal to the trainer.

    Args:
        monitor: the name of the value to track.
        min_delta: the minimum difference between losses to increase the patience counter.
        patience: the number of times to wait until the trainer does not terminate.
        max_mode: if true metric will be multiply by -1,
                  turn this flag when increasing metric value is expected for example Accuracy

    **Usage example:**

    .. code:: python
        1. Define the callback
        early_stop = EarlyStopping(
            monitor="loss", patience=10
        )
            
        2. Add the callback to the trainer
        trainer = NewTrainer(
            callbacks={"on_epoch_end", early_stop}
        )
        This will replace the default on_epoch_end callback with the early_stop callback.
        Following the same pattern, you can replace any callback in white list.

    """

    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 8,
        max_mode: bool = False,
    ) -> None:
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        # flag to reverse metric, for example in case of accuracy metric where bigger value is better
        # In classical loss functions smaller value = better,
        # in case of max_mode training end with metric stable/decreasing
        self.max_mode = max_mode

        self.counter: int = 0
        self.best_score: float = -inf if max_mode else inf
        self.early_stop: bool = False

    def __call__(self, model: Module, epoch: int,
                 valid_metric: Dict[str, AverageMeter]) -> TrainerState:
        score: float = valid_metric[self.monitor].avg
        is_best: bool = score > self.best_score if self.max_mode else score < self.best_score
        if is_best:
            self.best_score = score
            self.counter = 0
        else:
            # Example score = 1.9 best_score = 2.0 min_delta = 0.15
            # with max_mode (1.9 > (2.0 - 0.15)) == True
            # with min_mode (1.9 < (2.0 + 0.15)) == True
            is_within_delta: bool = (score > (self.best_score -
                                              self.min_delta) if self.max_mode
                                     else score < (self.best_score +
                                                   self.min_delta))
            if not is_within_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

        if self.early_stop:
            print(
                f"[INFO] Early-Stopping the training process. Epoch: {epoch}.")
            return TrainerState.TERMINATE

        return TrainerState.TRAINING


class SaveCheckpoint:
    '''
    every interval epochs save the model state to the path.
    '''

    def __init__(self, interval: int, path: Union[str, Path]):
        self.interval = interval
        self.path = path

    def __call__(self, model: Module, epoch: int, accelerator: Accelerator,
                 valid_stats: Dict[str, AverageMeter]):
        if epoch == 1 or epoch % self.interval == 0:
            path = self.path.format(epoch)
            os.makedirs(path, exist_ok=True)
            accelerator.save_state(path)


class SaveModel:
    '''
    Save the model to the path.
    '''

    def __init__(self, path: Union[str, Path]):
        self.path = path

    def __call__(self, model: Module, accelerator: Accelerator,
                 valid_stats: Dict[str, AverageMeter]):
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save_model(unwrapped_model, self.path)
