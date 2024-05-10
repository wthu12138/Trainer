from trainer_template import Trainer
from accelerate import Accelerator
from torch.utils.data import DataLoader
from model import ToyModel
from dataset import ToyDataset
from torch import optim
from trainer_utils import get_args, load_config
import torch
from callback import SaveCheckpoint, SaveModel


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
        super().__init__(model, train_dataloader, valid_dataloader, criterion,
                         optimizer, scheduler, accelerator, training_config,
                         saveing_config, callbacks)
        '''
        May change the training loop here, redefine the fit_epoch method
        '''


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

    # Define the callbacks
    save_checkpoint = SaveCheckpoint(interval=2,
                                     path=saving_config['checkpoint_path'])
    save_model = SaveModel(path=saving_config['model_save_path'])

    # Initialize the trainer
    trainer = CustomTrainer(model=model,
                            train_dataloader=train_data_loader,
                            valid_dataloader=valid_data_loader,
                            criterion=mse_func,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            accelerator=accelerator,
                            training_config=training_config,
                            saveing_config=saving_config,
                            callbacks={
                                "on_checkpoint": save_checkpoint,
                                'after_fit': save_model
                            })
    #train the model
    trainer.fit()


if __name__ == '__main__':
    main()
