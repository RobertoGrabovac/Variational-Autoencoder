import os
import matplotlib.pyplot as plt

from torch import optim
from torch import Tensor
import pytorch_lightning as pl
import torchvision.utils as vutils

from models.vae import VAE


class Experiment(pl.LightningModule):

    def __init__(self, vae_model: VAE, params: dict) -> None:
        super(Experiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False # do not retain the autograd graph after backward pass (frees memory). 
                                # Set to True if you need to perform multiple backward passes on the
                                # same graph (e.g., when handling multiple loss components). 

        self.training_losses = []  
        self.validation_losses = []
        self.epoch_validation_losses = []  
        self.reconstruction_losses = []
        self.kld_losses = []

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    # those are the default parameters of training_step method. Parameter optimizer_idx is set automatically 
    # by Lightning when using multiple optimizers which is crucial for architectures with multiple optimizers
    # (GANs). If we have two optimizers, Lightning will execute training_step twice for each batch: once with
    # optimizer_idx=0 (for the first optimizer) and once with optimizer_idx=1 (for the second optimizer).
    def training_step(self, batch, batch_idx, optimizer_idx=None):
        real_img, _ = batch
        self.curr_device = real_img.device # automatically set up by Trainer based on accelerator and devices parameters

        results = self.forward(real_img)
        train_loss = self.model.loss_function(*results, kld_weight=self.params['kld_weight_train'])

        self.training_losses.append(train_loss['loss'].item())
        self.reconstruction_losses.append(train_loss['Reconstruction_Loss'].item())
        self.kld_losses.append(train_loss['KLD'].item())

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        # after the return of train_loss, backpropagation is automatically executed
        return train_loss['loss']

    # same reasoning as for the training_step method
    def validation_step(self, batch, batch_idx, optimizer_idx=None):
        real_img, _ = batch
        self.curr_device = real_img.device

        results = self.forward(real_img)
        val_loss = self.model.loss_function(*results, kld_weight=self.params['kld_weight_val'])

        self.validation_losses.append(val_loss['loss'].item())
        
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        
        return val_loss['loss']

    def on_train_end(self) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses, label="Training Loss", color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid(False)
        
        plt.yscale("log")

        plot_path_train = os.path.join("/kaggle/working", "training_loss.png")
        plt.savefig(plot_path_train)
        print(f"Training Loss plot saved at {plot_path_train}")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.epoch_validation_losses, label="Validation Loss", color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Validation Loss (Per Epoch)")
        plt.legend()
        plt.grid(False)

        plot_path_val = os.path.join("/kaggle/working", "validation_loss_epoch.png")
        plt.savefig(plot_path_val)  
        print(f"Validation Loss plot saved at {plot_path_val}")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.validation_losses, label="Validation Loss", color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.legend()
        plt.grid(False)
        
        plot_path_val = os.path.join("/kaggle/working", "validation_loss.png")
        plt.savefig(plot_path_val)
        print(f"Validation Loss plot saved at {plot_path_val}")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.reconstruction_losses, label="Reconstruction Loss", color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Reconstruction Loss")
        plt.legend()
        plt.grid(False)
        
        plot_path_recons = os.path.join("/kaggle/working", "reconstruction_loss.png")
        plt.savefig(plot_path_recons)
        print(f"Reconstruction Loss plot saved at {plot_path_recons}")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.kld_losses, label="KLD Loss", color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("KLD Loss")
        plt.legend()
        plt.grid(False)
        
        plot_path_kld = os.path.join("/kaggle/working", "kld_loss.png")
        plt.savefig(plot_path_kld)
        print(f"KLD Loss plot saved at {plot_path_kld}")
        plt.close()
        
    def on_validation_end(self) -> None:
        self.sample_images()

    def on_validation_epoch_end(self):
        if len(self.validation_losses) > 0:
            final_val_loss = sum(self.validation_losses) / len(self.validation_losses)
            self.epoch_validation_losses.append(final_val_loss)  
            self.log("val_loss", final_val_loss, sync_dist=True)  
            self.validation_losses.clear()  
        
    def sample_images(self):
        test_input, _ = next(iter(self.trainer.datamodule.test_dataloader()))

        # select the first 36 test images (to form a 6x6 grid) and move them to the current device
        # to generate their reconstructions
        test_input = test_input[:36].to(self.curr_device) 

        recons = self.model.generate(test_input)
                
        vutils.save_image(test_input.data,
                        os.path.join(self.logger.log_dir,
                                    "Reconstructions",
                                    f"TEST_INPUT_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                        normalize=True, # note that we used tanh in the final layer of decoder which actually produces
                        nrow=6)         # values between -1 and 1, even though our dataset pixels range from 0 to 1.
                                        # This normalize=True normalizes the input images to the [0, 1] range.
        vutils.save_image(recons.data,
                        os.path.join(self.logger.log_dir,
                                    "Reconstructions",
                                    f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                        normalize=True,
                        nrow=6)

        try:
            # generate 36 sample images on the current device; this number is chosen to create a 6x6 grid (using nrow=6)
            samples = self.model.sample(36, self.curr_device) 
            vutils.save_image(samples.cpu().data,
                            os.path.join(self.logger.log_dir,
                                        "Samples",
                                        f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                            normalize=True,
                            nrow=6)
        except Warning:
            pass

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               betas=(self.params['beta_1'], self.params['beta_2']),
                               weight_decay=self.params['lambda_l2'])
        optims.append(optimizer)

        if self.params['scheduler_gamma'] is not None:
            # scheduler systematically changes the learning rate over the course of training, 
            # typically decreasing it over time. This helps the model make large updates early 
            # on (to quickly converge) and smaller, more precise updates later (to fine-tune the solution).
            # Other types: StepLR, CosineAnnealingLR
            scheduler = optim.lr_scheduler.ExponentialLR(optims[0], gamma=self.params['scheduler_gamma'])
            scheds.append(scheduler)

            return optims, scheds
        return optims