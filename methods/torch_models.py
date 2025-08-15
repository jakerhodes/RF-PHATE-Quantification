import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class LinearActivation(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob=0):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.elu(self.linear(x))
        x = self.dropout(x)
        return x

class LinearBlock(nn.Sequential):
    def __init__(self, dim_list, dropout_prob=0):
        modules = [LinearActivation(dim_list[i - 1], dim_list[i], dropout_prob) for i in range(1, len(dim_list) - 1)]
        modules.append(nn.Linear(dim_list[-2], dim_list[-1]))  # No activation for the last layer
        super().__init__(*modules)

class AETorchModule(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim, dropout_prob=0):
        super().__init__()

        full_list = [input_dim] + list(hidden_dims) + [z_dim]
        self.encoder = LinearBlock(dim_list=full_list, dropout_prob=dropout_prob)

        full_list.reverse()
        full_list[0] = z_dim
        self.decoder = LinearBlock(dim_list=full_list, dropout_prob=dropout_prob)

    def forward(self, x):
        z = self.encoder(x)
        z_decoder = z
        recon = self.decoder(z_decoder)
        return recon, z


class ProxAETorchModule(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim, output_activation='log_softmax', recon_dim=None, dropout_prob=0):
        """
        Args:
            input_dim (int): Dimension of input data
            hidden_dims (list of int): List of hidden layer dimensions
            z_dim (int): Latent space dimension
            output_activation (str): Type of output activation ('log_softmax', 'none', etc.)
            dropout_prob (float): Dropout probability
            recon_dim (int, optional): Dimension of reconstruction output. Defaults to input_dim.
        """
        super().__init__()

        if recon_dim is None:
            recon_dim = input_dim

        # Encoder
        full_list = [input_dim] + list(hidden_dims) + [z_dim]
        self.encoder = LinearBlock(dim_list=full_list, dropout_prob=dropout_prob)

        # Decoder
        decoder_list = [z_dim] + list(reversed(hidden_dims)) + [recon_dim]
        self.decoder = LinearBlock(dim_list=decoder_list, dropout_prob=dropout_prob)

        # Output activation
        self.output_activation = output_activation
        if output_activation == 'log_softmax':
            self.final_activation = nn.LogSoftmax(dim=1)
        elif output_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif output_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif output_activation == 'none':
            self.final_activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported output_activation: {output_activation}")


    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        recon = self.final_activation(recon)
        return recon, z

class MLPReg(nn.Module):
    def __init__(self, encoder, input_dim, output_dim):
        super().__init__()

        self.encoder = encoder
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        z = self.encoder(x)
        pred = self.linear(z)
        return pred
    
class EarlyStopping:
    def __init__(self, patience=5, delta_factor=0.01, save_model=False, save_path="best_model.pth"):
        self.patience = patience
        self.delta_factor = delta_factor   # Set delta as a percentage of loss
        self.save_model = save_model
        self.save_path = save_path

    def __call__(self, current_loss, best_loss,counter, model):
        # Save the best model if the current loss is the lowest so far
        if best_loss is None:
            best_loss = current_loss   # first epoch

        dynamic_delta = best_loss * self.delta_factor

        if best_loss - current_loss > dynamic_delta:
            counter = 0  # Reset patience counter because there's improvement
        else:
            counter += 1  # No significant improvement, increase counter

        # save model if the current loss is the lowest so far
        if best_loss > current_loss:
            best_loss = current_loss
            if self.save_model: 
                torch.save(model, self.save_path)
        
        return counter >= self.patience, best_loss, counter  # Stop training if patience is exceeded