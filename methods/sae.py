from sklearn.base import BaseEstimator, TransformerMixin
from methods.nn.train import train_sae
import methods.nn.models as models
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class SAE(BaseEstimator, TransformerMixin):

    def __init__(self, input_dim,
        hidden_dim1 = 800, 
        hidden_dim2 = 400,
        hidden_dim3 = 100,
        z_dim = 2,
        epochs = 50,
        device = None,
        lr = 0.01,
        batch_size = 16,
        num_workers = 4,
        num_classes = 10,
        alpha = 0.5,
        random_state = None

      ):
        self.input_dim   = input_dim # Maybe don't need this until fitting; infer from data
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim1 = hidden_dim2
        self.hidden_dim1 = hidden_dim3
        self.z_dim = z_dim
        self.epochs = epochs
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes # Maybe also infer from data
        self.alpha = alpha

        self.encoder = models.Encoder(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, z_dim)
        self.decoder = models.Decoder(z_dim, input_dim, hidden_dim3, hidden_dim2, hidden_dim1)
        self.head = models.ClassificationHead(self.encoder, input_dim = z_dim, hidden_dim1 = 100,
            hidden_dim2 = 50, output_dim = num_classes) # may want to make this flexible
        self.sae = models.SupervisedAE(self.encoder, self.decoder, self.head)
        self.criterion_recon = torch.nn.MSELoss()
        self.criterion_pred = torch.nn.CrossEntropyLoss()
        self.optimizer  = torch.optim.Adam(params = self.sae.parameters(), lr = lr)
        self.random_state = random_state
        
    def fit(self, X, y):
        self._fit_transform(X, y)


    def _fit_transform(self, X, y):

        if self.device is None:
            # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = 'cpu'

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        y = torch.tensor(y, dtype = torch.long) # May have to double check type here
        X = torch.tensor(X, dtype = torch.float)
        X = X.to(torch.device(self.device))

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers)
        
        # Come back here after looking over train_sae file
        model = train_sae(self.sae, self.epochs, dataloader, self.optimizer, self.criterion_pred,
            self.criterion_recon, self.alpha, self.device)

        self.embedding_ = model.Encoder(X).detach().numpy()

    def fit_transform(self, X, y):
        self._fit_transform(X, y)
        return self.embedding_
