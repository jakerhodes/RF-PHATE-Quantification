import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from methods.utils.numpy_dataset import FromNumpyDataset
from methods.base_model import BaseModel
from methods.torch_models import AETorchModule, EarlyStopping
import numpy as np
# import wandb
import logging

class CE(BaseModel):
    """
    Centroid Autoencoder.
    """

    def __init__(self,
                 n_components=2,
                 lr=1e-3,
                 batch_size=512,
                 weight_decay=1e-5,
                 random_state=None,
                 device=None,
                 dropout_prob=0.0,
                 epochs=200,
                 hidden_dims=[800, 400, 100],
                 early_stopping=False,
                 patience=50,
                 delta_factor=1e-3,
                 save_model=False
                 ):

        self.n_components = n_components
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = device
        self.dropout_prob = dropout_prob
        self.epochs = epochs
        self.data_shape = None
        self.hidden_dims = hidden_dims
        self.centroids = None
        self.early_stopping = early_stopping
        self.patience = patience
        self.delta_factor = delta_factor
        self.save_model = save_model


    def init_torch_module(self, data_shape):

        input_size = data_shape

        self.torch_module = AETorchModule(input_dim   = input_size,
                                          hidden_dims = self.hidden_dims,
                                          z_dim       = self.n_components,
                                          dropout_prob=self.dropout_prob)

    def compute_centroids(self, x, y):
        """
        Compute and store the centroids for each class.
        """
        unique_classes = torch.unique(y)
        self.centroids = {cls.item(): x[y == cls].mean(dim=0) for cls in unique_classes}

    def fit(self, x, y):

        self.data_shape = x.shape[1]

        tensor_dataset = TensorDataset(
            torch.tensor(x, dtype=torch.float),
            torch.tensor(y, dtype=torch.long)
        )

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


        self.init_torch_module(self.data_shape)
        self.optimizer = torch.optim.AdamW(self.torch_module.parameters(),
                                            lr=self.lr,
                                            weight_decay=self.weight_decay)
        self.criterion = nn.MSELoss(reduction='none')  # Allow per-instance loss computation

        # Compute centroids once
        x_tensor = torch.tensor(x, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.long)
        self.compute_centroids(x_tensor, y_tensor)

        train_loader = DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_loop(self.torch_module, self.epochs, train_loader, self.optimizer, self.device) 

    def compute_loss(self, y, x_hat):
        """
        Compute the centroid-based reconstruction loss using precomputed centroids.
        """
        # Map each instance to its corresponding centroid
        centroid_targets = torch.stack([self.centroids[cls.item()] for cls in y]).to(self.device)

        # Compute the loss, as define in the CE paper
        loss = 0.5 * ((x_hat - centroid_targets) ** 2).sum(dim=1).mean()

        self.recon_loss_temp = loss.item()
        return loss


    def train_loop(self, model, epochs, train_loader, optimizer, device='cpu'):

        self.epoch_losses_recon = []
        best_loss = float("inf")
        counter=0
        
        for _, epoch in enumerate(range(epochs)):

            model = model.train()
            model = model.to(device)

            running_recon_loss = 0

            for _, (x, y) in enumerate(train_loader, 0):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                recon, _ = model(x)

                self.compute_loss(y, recon).backward()

                running_recon_loss += self.recon_loss_temp

                optimizer.step()

            self.epoch_losses_recon.append(running_recon_loss / len(train_loader))
            # wandb.log({f"{self.random_state}: train_recon_loss": self.epoch_losses_recon[-1], "Epoch": epoch})

            if epoch % 50 == 0:
                logging.info(f"Epoch {epoch}/{self.epochs}, Recon Loss: {self.epoch_losses_recon[-1]:.7f}") 


            # Check for early stopping
            if self.early_stopping:
                os.makedirs(f"{self.random_state}/", exist_ok=True)
                subfolder_path = f"{self.random_state}/best_{self.random_state}.pth"
                early_stopping = EarlyStopping(patience = self.patience,
                                        delta_factor = self.delta_factor, 
                                        save_model = self.save_model, 
                                        save_path = subfolder_path)
                should_stop, best_loss, counter = early_stopping(self.epoch_losses_recon[-1], best_loss, counter, model)
                if should_stop:
                    logging.info(f"Stopping training early at epoch {epoch}")
                    return  

    def transform(self, x):
        self.torch_module.eval()

        x = TensorDataset(torch.tensor(x, dtype=torch.float))

        loader = DataLoader(x, batch_size=self.batch_size, shuffle=False)
 
        z = [self.torch_module.encoder(batch[0].to(self.device)).cpu().detach().numpy() for batch in loader]
        return np.concatenate(z)

    def inverse_transform(self, x):
        self.torch_module.eval()
        x = FromNumpyDataset(x)
        loader = DataLoader(x, batch_size=self.batch_size, shuffle=False)
        x_hat = [self.torch_module.decoder(batch.to(self.device)).cpu().detach().numpy()
                 for batch in loader]

        return np.concatenate(x_hat)