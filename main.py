import os
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from models import *
from experiment import Experiment
from dataset import Dataset

from utils import parse_args, load_model_config, initialize_models, count_parameters


args = parse_args()
config = load_model_config(args.filename)
vae_models = initialize_models()


def main():
    model = vae_models[config['model_params']['name']](**config['model_params'])
    total_params = count_parameters(model)

    experiment = Experiment(model, config['exp_params'])

    data = Dataset(**config["data_params"], pin_memory=len(config['trainer_params']['devices']) != 0)
    data.setup()

    # also creates version folders like save_dir/name/version_i where i is the number of experiment
    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=config['model_params']['name'])
    
    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True) # new images produced by generative model after each epoch
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True) # training images for validation of generative 
                                                                                    # model after each epoch
    
    runner = Trainer(logger=tb_logger,
                     callbacks=[
                         LearningRateMonitor(),
                         ModelCheckpoint(save_top_k=2, 
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"), # can't open in Kaggle
                                         monitor="val_loss",
                                         save_last=True)
                     ],
                     accelerator=config['trainer_params']['accelerator'],  
                     devices=config['trainer_params']['devices'],  
                     max_epochs=config['trainer_params']['max_epochs'],
                     strategy=DDPStrategy(find_unused_parameters=False)) # this parameter only makes sense for multi GPU.
                                                                         # Our model does not have unused parameters
                                                                         # so by setting find_unused_parameters=False 
                                                                         # we speed up the training process. Some models
                                                                         # might include layers that aren’t always used 
                                                                         # in every forward pass (used only if some conditions
                                                                         # are met), so we set find_unused_parameters=True

    print(f"======= Training {config['model_params']['name']} =======")
    print(f"======= Total params: {total_params} =======")

    runner.fit(experiment, datamodule=data)

if __name__ == '__main__':
    main()