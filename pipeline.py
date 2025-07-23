import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import numpy as np
import inspect




# ----------- Modelo MLP -----------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout, n_classes):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
def get_sample_weights_loss(y):
  y = np.asarray(y, dtype=np.int64)
  class_counts = np.bincount(y)
  class_weights = 1.0 / class_counts
  class_weights = class_weights / class_weights.sum()

  return class_weights

def Standard_scaler_channel(X_train, X_test):
    def scale(X):
        if isinstance(X, torch.Tensor):
            mean = X.mean(dim=(1, 2), keepdim=True)
            std = X.std(dim=(1, 2), keepdim=True)
            return (X - mean) / (std + 1e-8)
        elif isinstance(X, np.ndarray):
            mean = np.mean(X, axis=(1, 2), keepdims=True)
            std = np.std(X, axis=(1, 2), keepdims=True)
            return (X - mean) / (std + 1e-8)
        else:
            raise TypeError("Input debe ser torch.Tensor o np.ndarray")

    return scale(X_train), scale(X_test)

def MinMax_scaler_channel(X_train, X_test):
    def scale(X):
        if isinstance(X, torch.Tensor):
            X_min = X.amin(dim=(1, 2), keepdim=True)
            X_max = X.amax(dim=(1, 2), keepdim=True)
            return (X - X_min) / (X_max - X_min + 1e-8)
        elif isinstance(X, np.ndarray):
            X_min = np.min(X, axis=(1, 2), keepdims=True)
            X_max = np.max(X, axis=(1, 2), keepdims=True)
            return (X - X_min) / (X_max - X_min + 1e-8)
        else:
            raise TypeError("Input debe ser torch.Tensor o np.ndarray")

    return scale(X_train), scale(X_test)

class Pytorch_Pipeline():
      def __init__(self, model_class, sample_weights_loss=None, max_epochs = 200):
        self.model_class = model_class
        self.model = None
        self.params = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_weights_loss = sample_weights_loss
        self.criterion = None
        self.optimizer = None
        self.batch_size = None
        self.max_epochs = max_epochs
        self.scaler = None


      def partial_fit(self, loader):
        self.model.to(self.device)
        self.model.train()

        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(xb), yb)
            loss.backward()
            self.optimizer.step()

        return self

      def predict(self, X):
          self.model.eval()
          if self.scaler is not None:
            X = self.scaler.transform(X)
          X_tensor = torch.tensor(X, dtype=torch.float32)
          dataset = TensorDataset(X_tensor)
          loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

          preds = []
          with torch.no_grad():
              for xb in loader:
                  xb = xb[0].to(self.device)
                  pred = self.model(xb)
                  preds.append(pred.cpu().numpy())
          preds = np.concatenate(preds, axis=0)
          return np.argmax(preds, axis=1)

      def predict_and_evaluate(self, loader):
          self.model.eval()
          val_loss, n_samples = 0.0, 0
          all_preds, all_targets = [], []
          with torch.no_grad():
              for xb, yb in loader:
                  xb, yb = xb.to(self.device), yb.to(self.device)
                  output = self.model(xb)
                  loss = self.criterion(output, yb)
                  val_loss += self.criterion(output, yb).item() * xb.size(0)
                  n_samples += xb.size(0)

                  pred = output.argmax(dim=1)
                  all_preds.append(pred.cpu())
                  all_targets.append(yb.cpu())

          avg_val_loss = val_loss / n_samples
          y_true = torch.cat(all_targets).numpy()
          y_pred = torch.cat(all_preds).numpy()
          f1 = f1_score(y_true, y_pred, average='weighted')  # weighted F1

          return avg_val_loss, f1, y_true, y_pred

      def set_params(self, **params):
          self.params = params

          # Obtener los parámetros esperados por el constructor de model_class
          signature = inspect.signature(self.model_class.__init__)
          valid_keys = set(signature.parameters.keys()) - {'self'}

          # Filtrar los params para incluir solo los esperados
          filtered_params = {k: v for k, v in params.items() if k in valid_keys}

          self.model = self.model_class(**filtered_params)
          self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])
          self.batch_size = self.params['batch_size']

      def set_criterion(self, y):
          # ----------- Criterion -----------
          if self.sample_weights_loss is not None:
              class_weights = get_sample_weights_loss(y)
              class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
              self.criterion = nn.CrossEntropyLoss(weight=class_weights)
              print('Using class weight')
          else:
              self.criterion = nn.CrossEntropyLoss()

          return self

      def set_scaler_transform(self, scaler, X_train, X_test, dtype = 'Tabular'):
          # ----------- Escalado de datos -----------
          if dtype == 'Tabular':
            if scaler == 'standard':
                self.scaler = StandardScaler()
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)
            elif scaler == 'minmax':
                self.scaler = MinMaxScaler()
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)
            else: pass


          elif dtype == 'MultiDim_TimeSeries':
            if scaler == 'standard':
                X_train, X_test = Standard_scaler_channel(X_train, X_test)
            elif scaler == 'minmax':
                X_train, X_test = MinMax_scaler_channel(X_train, X_test)
            else: pass

          return X_train, X_test


      def set_scaler(self, scaler):
          # ----------- Escalado de datos -----------
          if scaler == 'standard':
              self.scaler = StandardScaler()

          elif scaler == 'minmax':
              self.scaler = MinMaxScaler()

          else:
              return None

      def fit_early_stopping(self, X_train, y_train, X_test, y_test, scaler = 'standard'):

          X_train, X_test = self.set_scaler_transform(scaler, X_train, X_test)

          X_train = torch.tensor(X_train, dtype=torch.float32)
          y_train = torch.tensor(y_train, dtype=torch.long)
          train_dataset = TensorDataset(X_train, y_train)
          train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

          X_test = torch.tensor(X_test, dtype=torch.float32)
          y_test = torch.tensor(y_test, dtype=torch.long)
          test_dataset = TensorDataset(X_test, y_test)
          test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

          self.set_criterion(y_train)

          for epoch in range(self.max_epochs):
              self.partial_fit(train_loader)
              avg_val_loss, f1, _, _ = self.predict_and_evaluate(test_loader)
              # ---------- Early stopping (por pérdida) ----------
              patience = 10
              min_delta = 1e-4
              best_val_loss = float('inf')
              epochs_no_improve = 0
              best_model_state = None

              if avg_val_loss + min_delta < best_val_loss:
                  best_val_loss = avg_val_loss
                  best_model_state = self.model.state_dict()
                  epochs_no_improve = 0
              else:
                  epochs_no_improve += 1
                  if epochs_no_improve >= patience:
                      break

          return f1
