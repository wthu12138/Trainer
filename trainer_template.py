from typing import Any, Callable, Dict, Optional
import torch
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader
from omegaconf import DictConfig, ListConfig
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from accelerate import Accelerator
from trainer_utils import *

callbacks_whitelist = [
    # high level functions
    "preprocess",
    "augmentations",
    "evaluate",
    "fit",
    "fit_epoch",
    # events (by calling order)
    "on_epoch_start",
    "on_before_model",
    "on_after_model",
    "on_checkpoint",
    "on_epoch_end",
]


class Trainer:
    """
    Args:
        model: the nn.Module to be optimized.
        train_dataloader: the data loader used in the training loop.
        valid_dataloader: the data loader used in the validation loop.
        criterion: the nn.Module with the function that computes the loss.
        optimizer: the torch optimizer object to be used during the optimization.
        scheduler: the torch scheduler object with defiing the scheduling strategy.
        accelerator: the Accelerator object to distribute the training.
        training_config: a TrainerConfiguration structure containing the experiment hyper parameters.
        saveing_config: a dictionary containing the configuration to save the model.
        callbacks: a dictionary containing the pointers to the functions to overrides. The
          main supported hooks are ``evaluate``, ``preprocess``, ``augmentations`` and ``fit``.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader[Any],
        valid_dataloader: DataLoader[Any],
        criterion: Optional[nn.Module],
        optimizer: Optimizer,
        scheduler: lr_scheduler._LRScheduler,
        accelerator: Accelerator,
        training_config: Union[DictConfig, ListConfig],
        saveing_config: Optional[Union[DictConfig, ListConfig]],
        callbacks: Dict[str, Callable[..., None]] = {},
    ) -> None:
        # setup the accelerator
        if Accelerator is None:
            raise ModuleNotFoundError('accelerate library is not installed')
        self.accelerator = accelerator
        self.model = self.accelerator.prepare(model)
        self.train_dataloader = self.accelerator.prepare(train_dataloader)
        self.valid_dataloader = self.accelerator.prepare(valid_dataloader)
        self.criterion = None if criterion is None else criterion.to(
            self.device)
        self.optimizer = self.accelerator.prepare(optimizer)
        self.scheduler = self.accelerator.prepare(scheduler)
        self.training_config = training_config
        self.saveing_config = saveing_config
        self.num_epochs = training_config['epochs']
        self.state = TrainerState.STARTING
        self.global_step = 0

        # configure callbacks
        for fn_name, fn in callbacks.items():
            if fn_name not in callbacks_whitelist:
                raise ValueError(f"Not supported: {fn_name}.")
            setattr(Trainer, fn_name, fn)

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    def backward(self, loss: Tensor) -> None:
        self.accelerator.backward(loss)

    def fit_epoch(self, epoch: int) -> None:
        self.model.train()
        losses = AverageMeter()
        for sample_id, sample in enumerate(self.train_dataloader):
            sample = {"input": sample[0], "target": sample[1]}
            self.optimizer.zero_grad()

            sample = self.preprocess(sample)
            sample = self.augmentations(sample)
            sample = self.on_before_model(sample)

            output = self.on_model(self.model, sample)
            self.on_after_model(output, sample)
            loss = self.compute_loss(output, sample["target"])

            self.backward(loss)
            self.optimizer.step()

            loss = self.accelerator.reduce(loss, reduction='mean')

            self.global_step += 1
            if self.accelerator.is_main_process:
                losses.update(loss.item(), len(sample["input"]))
                if sample_id % 50 == 0:
                    self.accelerator.log({"train_loss": losses.avg},
                                         step=self.global_step)

    def fit(self) -> None:
        bar = tqdm(range(1, self.num_epochs + 1),
                   disable=not self.accelerator.is_main_process)
        for epoch in bar:
            self.state = TrainerState.TRAINING
            self.fit_epoch(epoch)
            self.state = TrainerState.VALIDATE
            valid_stats = self.evaluate()
            self.on_checkpoint(self.model, epoch, valid_stats)
            self.on_epoch_end()
            if self.state == TrainerState.TERMINATE:
                break
            self.scheduler.step()
            bar.update(1)
        self.after_fit()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, AverageMeter]:
        self.model.eval()
        stats = StatsTracker()
        for sample_id, sample in enumerate(self.valid_dataloader):
            sample = {"input": sample[0], "target": sample[1]}

            sample = self.preprocess(sample)
            sample = self.on_before_model(sample)

            out = self.on_model(self.model, sample)
            self.on_after_model(out, sample)
            batch_size: int = len(sample["input"])
            val_loss = self.compute_loss(out, sample["target"])

            val_loss = self.accelerator.reduce(val_loss, reduction='mean')

            targets = self.accelerator.gather_for_metrics(sample["target"])
            out = self.accelerator.gather_for_metrics(out)

            if self.accelerator.is_main_process:
                stats.update_from_dict(self.compute_metrics(out, targets),
                                       batch_size)
                stats.update("val_loss", val_loss.item(), batch_size)
                if sample_id % 10 == 0:
                    self.accelerator.log(
                        {
                            key: meter.avg
                            for key, meter in stats.as_dict().items()
                        },
                        step=self.global_step)

        return stats.as_dict()

    def on_epoch_start(self, *args: Any, **kwargs: Any) -> None:
        ...

    def preprocess(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return x

    def augmentations(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return x

    def compute_metrics(self, *args: Any) -> Dict[str, float]:
        """Compute metrics during the evaluation."""
        return {}

    def compute_loss(self, *args: Tensor) -> Tensor:
        if self.criterion is None:
            raise RuntimeError("`criterion` should not be None.")
        return self.criterion(*args)

    def on_before_model(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return x

    def on_model(self, model: nn.Module, sample: Dict[str, Tensor]) -> Tensor:
        return model(sample["input"])

    def on_after_model(self, output: Tensor, sample: Dict[str,
                                                          Tensor]) -> None:
        ...

    def on_checkpoint(self, *args: Any, **kwargs: Dict[str, Any]) -> None:
        model, epoch, valid_stats = args
        if epoch == 1 or epoch % 10 == 0:
            self.accelerator.save_state(
                self.saveing_config['checkpoint_path'].format(epoch))

    def on_epoch_end(self, *args: Any, **kwargs: Dict[str, Any]) -> None:
        ...

    def after_fit(self, *args: Any, **kwargs: Dict[str, Any]) -> None:
        ...
