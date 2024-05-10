from trainer_template import Trainer
from accelerate import Accelerator
from typing import Dict, Any
from torch.utils.data import DataLoader
from model import ToyModel
from dataset import ToyDataset
from torch import optim
from trainer_utils import get_args, load_config
import torch


class CustomTrainer(Trainer):

    def __init__(self,
                 model,
                 train_dataloader,
                 valid_dataloader,
                 criterion,
                 optimizer,
                 scheduler,
                 accelerator,
                 training_config,
                 saveing_config,
                 callbacks={}):
        super().__init__(model,
                         train_dataloader,
                         valid_dataloader,
                         criterion,
                         optimizer,
                         scheduler,
                         accelerator,
                         training_config,
                         saveing_config,
                         callbacks={})

    #redefine the fit method
    def after_fit(self, *args: Any, **kwargs: Dict[str, Any]):
        self.accelerator.end_training()
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save_model(unwrapped_model,
                                    self.saveing_config['model_save_path'])


def main():
    args = get_args()
    config = load_config(args.file)

    # Extract the configuration
    experiment_config = config['experiment_settings']
    training_config = config['training_settings']
    saving_config = config['save_settings']

    # Initialize everything
    accelerator = Accelerator(log_with=experiment_config["log"])
    accelerator.init_trackers(project_name=experiment_config['project'],
                              config=dict(training_config))
    train_dataset = ToyDataset(length=1000)
    valid_dataset = ToyDataset(length=100)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   shuffle=True,
                                   batch_size=training_config['batch_size'])
    valid_data_loader = DataLoader(dataset=valid_dataset,
                                   shuffle=False,
                                   batch_size=training_config['batch_size'])
    model = ToyModel()
    optimizer = optim.Adam(model.parameters(),
                           lr=training_config['learning_rate'])
    mse_func = torch.nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=training_config['steplr_size'],
        gamma=training_config['steplr_gamma'])

    # Initialize the trainer
    trainer = CustomTrainer(model=model,
                            train_dataloader=train_data_loader,
                            valid_dataloader=valid_data_loader,
                            criterion=mse_func,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            accelerator=accelerator,
                            training_config=training_config,
                            saveing_config=saving_config)
    #train the model
    trainer.fit()


if __name__ == '__main__':
    main()
