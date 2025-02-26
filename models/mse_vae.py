import torch
from models.vae import VAE
from torch import nn
from torch.nn import functional as F
from typing import List
from torch import Tensor


class MSEVAE(VAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(MSEVAE, self).__init__()

        self.latent_dim = latent_dim

        # ENCODER
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # formula: floor(\frac{W + 2P - K}{S} + 1), where W=input size (height/width),
        # K=kernel_size, S=stride, P=padding
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    # we explain nn.BatchNorm2d for the first conv. layer with h_dim = 32.
                    # nn.BatchNorm2d(32) normalizes each of the 32 channels of a 4D tensor (e.g., (batch_size, 32, H, W)) independently.
                    # For example, if the input tensor has shape (8, 32, 32, 32):
                    #   - For each channel, it computes the mean and variance over all 8*32*32 = 8192 values.
                    #   - Each element x in a channel is normalized as: 
                    #         x_normalized = (x - mean) / sqrt(variance + eps)
                    #   - Then, the normalized value is scaled and shifted by learnable parameters gamma and beta:
                    #         y = gamma * x_normalized + beta
                    # This process stabilizes and speeds up training by reducing internal covariate shift.
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        # now we form a feedforward neural network -- encoder
        self.encoder = nn.Sequential(*modules)
        # Summary of encoder transformations:
        # Input: 3 channels, 64x64; CelebA dataset
        # 1st Conv: (3, 64, 64) -> (32, 32, 32)  [kernel=3, stride=2, padding=1]
        # 2nd Conv: (32, 32, 32) -> (64, 16, 16)
        # 3rd Conv: (64, 16, 16) -> (128, 8, 8)
        # 4th Conv: (128, 8, 8) -> (256, 4, 4)
        # 5th Conv: (256, 4, 4) -> (512, 2, 2)

        # flattened feature vectors and fully connected layers 
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # DECODER -- we simply reverse the process that encoder does
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        modules = []
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3,
                                       stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3,
                                               stride=2, padding=1, output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
                            nn.Tanh()) # common activation function in final layer of VAE decoders and other
                                       # generative models
        # Summary of decoder transformations:
        # Input latent feature map: (512, 2, 2)  [after flattening and reshaping from the latent vector]
        # 1st Transposed Conv: (512, 2, 2) -> (256, 4, 4)  [kernel=3, stride=2, padding=1, output_padding=1]
        # 2nd Transposed Conv: (256, 4, 4) -> (128, 8, 8)
        # 3rd Transposed Conv: (128, 8, 8) -> (64, 16, 16)
        # 4th Transposed Conv: (64, 16, 16) -> (32, 32, 32)
        # Final Layer:
        #   First, a Transposed Conv: (32, 32, 32) -> (32, 64, 64)
        #   Then, a Conv2d: (32, 64, 64) -> (3, 64, 64) with Tanh activation to output pixel values in [-1, 1]
        # NOTE: in the final layer we didn't go directly (32, 32, 32) -> (3, 64, 64) in order to get smoother, more 
        #       detailed reconstructions compared to this single, abrupt upsampling step


    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        # Flatten the tensor starting from dimension 1:
        # The batch dimension (dimension 0) remains unchanged,
        # and all subsequent dimensions (e.g., channels, height, width) are merged into one.
        # This converts a tensor of shape (batch_size, channels, height, width)
        # into one of shape (batch_size, channels * height * width).
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        # We predict log variance (logvar) instead of variance directly because:
        # - It allows the network to output unconstrained real numbers while ensuring variance is positive via exp()
        # - It improves numerical stability and simplifies optimization
        # - It makes computing the standard deviation (std = exp(0.5 * logvar)) straightforward for the reparameterization trick
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        # Here, result.view(-1, 512, 2, 2) reshapes 'result' to have:
        # - Batch size inferred automatically (-1)
        # - 512 channels
        # - A spatial dimension of 2x2
        result = result.view(-1, 512, 2, 2)

        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['kld_weight'] # there is also alternative to introduce KL annealing
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss

        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device:int,
               **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]