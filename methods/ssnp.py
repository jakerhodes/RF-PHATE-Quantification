import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.models.base_model import BaseModel
from src.models.torch_models import AETorchModule, EarlyStopping
import numpy as np
import wandb
import logging


class AETorchModuleWithClassifier(AETorchModule):
    def __init__(self, input_dim, hidden_dims, z_dim, n_classes, dropout_prob=0):
        super().__init__(input_dim, hidden_dims, z_dim, dropout_prob)
        self.classifier = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        recon, z = super().forward(x)
        class_pred = self.classifier(recon)
        return recon, class_pred, z


class SSNP(BaseModel):
    """
    Self-Supervised Network Projection (Espadoto et al., 2021) implementation
    with class labels for supervised training. Combines BCE and CCE losses.
    """

    def __init__(self,
                 n_components,
                 lr,
                 batch_size,
                 weight_decay,
                 random_state,
                 device,
                 dropout_prob,
                 epochs,
                 hidden_dims,
                 early_stopping,
                 patience,
                 delta_factor,
                 save_model):
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
        self.early_stopping = early_stopping
        self.patience = patience
        self.delta_factor = delta_factor
        self.save_model = save_model

        self._normalization_applied = False
        self._x_min = None
        self._x_ptp = None

    def init_torch_module(self, data_shape, n_classes):
        self.torch_module = AETorchModuleWithClassifier(
            input_dim=data_shape,
            hidden_dims=self.hidden_dims,
            z_dim=self.n_components,
            n_classes=n_classes,
            dropout_prob=self.dropout_prob)

    def fit(self, x, y):
        if not self._is_normalized(x):
            logging.warning("Input features are not in [0, 1]. Applying min-max normalization.")
            self._x_min = np.min(x, axis=0)
            self._x_ptp = np.ptp(x, axis=0) + 1e-8  # for stability
            x = (x - self._x_min) / self._x_ptp
            self._normalization_applied = True
        else:
            self._normalization_applied = False

        self.data_shape = x.shape[1]
        self.n_classes = len(np.unique(y))

        tensor_dataset = TensorDataset(
            torch.tensor(x, dtype=torch.float),
            torch.tensor(y, dtype=torch.long)
        )

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.init_torch_module(self.data_shape, self.n_classes)

        self.optimizer = torch.optim.AdamW(self.torch_module.parameters(),
                                           lr=self.lr,
                                           weight_decay=self.weight_decay)

        train_loader = DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle=True)
        self.train_loop(self.torch_module, self.epochs, train_loader, self.optimizer, self.device)

    def compute_loss(self, x, x_hat, y, y_hat):
        bce_loss = nn.BCEWithLogitsLoss()(x_hat, x)
        cce_loss = nn.CrossEntropyLoss()(y_hat, y)
        loss = 0.5 * bce_loss + 0.5 * cce_loss  # lambda = 0.5 from SSNP paper
        self.loss_temp = loss.item()
        return loss

    def train_loop(self, model, epochs, train_loader, optimizer, device='cpu'):
        self.epoch_losses = []
        best_loss = float("inf")
        counter = 0

        for epoch in range(epochs):
            model.train().to(device)
            running_loss = 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                x_hat, y_hat, _ = model(x)
                loss = self.compute_loss(x, x_hat, y, y_hat)
                loss.backward()
                optimizer.step()
                running_loss += self.loss_temp

            avg_loss = running_loss / len(train_loader)
            self.epoch_losses.append(avg_loss)
            wandb.log({f"{self.random_state}: train_loss": avg_loss, "Epoch": epoch})

            if epoch % 50 == 0:
                logging.info(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.7f}")

            if self.early_stopping:
                os.makedirs(f"{self.random_state}/", exist_ok=True)
                subfolder_path = f"{self.random_state}/best_{self.random_state}.pth"
                early_stopping = EarlyStopping(patience=self.patience,
                                               delta_factor=self.delta_factor,
                                               save_model=self.save_model,
                                               save_path=subfolder_path)
                should_stop, best_loss, counter = early_stopping(avg_loss, best_loss, counter, model)
                if should_stop:
                    logging.info(f"Stopping training early at epoch {epoch}")
                    return

    def transform(self, x):
        if self._normalization_applied:
            if self._x_min is None or self._x_ptp is None:
                raise RuntimeError("Normalization was applied in fit, but parameters are missing.")
            x = (x - self._x_min) / self._x_ptp

        self.torch_module.eval()
        dataset = TensorDataset(torch.tensor(x, dtype=torch.float))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        embeddings = [
            self.torch_module.encoder(batch[0].to(self.device)).cpu().detach().numpy()
            for batch in loader
        ]
        return np.concatenate(embeddings)

    def _is_normalized(self, x):
        return np.all((x >= 0) & (x <= 1))