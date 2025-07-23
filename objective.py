import numpy as np
from collections import Counter
import copy
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import inspect
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pipeline import Pytorch_Pipeline, MLP


def get_metrics(y_true, y_pred):
  metrics = {
      'accuracy': accuracy_score(y_true, y_pred),
      'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
      'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
      'f1_score': f1_score(y_true, y_pred, average='weighted'),
      'cm': confusion_matrix(y_true, y_pred)
  }
  print(metrics)

class optuna_objective:
    def __init__(self, X_train, X_test, y_train, y_test, n_classes, SMOTE_on=None, sample_weights_loss=None):
        self.results = {}
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_classes = n_classes
        self.sample_weights_loss = sample_weights_loss
        self.max_epochs = 200
        self.best_model_trial = None

    def get_loaders(self, X_train, X_test, y_train, y_test, batch_size):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        return train_loader, test_loader

    def objective(self, trial):
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        # ----------- Hiperparámetros a optimizar -----------
        n_layers = trial.suggest_int("n_layers", 1, 4)
        params = {
            'hidden_sizes' : [trial.suggest_int(f"n_units_l{i}", 16, 528, step=64) for i in range(n_layers)],
            'dropout' : trial.suggest_float("dropout", 0.1, 0.5),
            'lr' : trial.suggest_float("lr", 1e-4, 5e-1, log=True),
            'batch_size' : trial.suggest_categorical("batch_size", [32, 64, 128]),
            'input_dim' : X_train.shape[1],
            'n_classes' : self.n_classes,
        }

        pipeline_mlp =  Pytorch_Pipeline(model_class=MLP, sample_weights_loss = self.sample_weights_loss)
        #Set params
        pipeline_mlp.set_params(**params)
        #Set criterion
        pipeline_mlp.set_criterion(y_train)

        # ----------- Escalado de datos -----------
        scaler = trial.suggest_categorical("scaler", ['None', 'standard', 'minmax'])
        X_train, X_test = pipeline_mlp.set_scaler_transform(scaler, X_train, X_test)
        # ------------- Loaders --------------------
        train_loader, test_loader = self.get_loaders(X_train, X_test, y_train, y_test, pipeline_mlp.batch_size)

        # ---------- Early stopping (por loss) ----------
        patience = 10
        min_delta = 1e-4
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(pipeline_mlp.max_epochs):
              pipeline_mlp.partial_fit(train_loader)
              avg_val_loss, f1, y_test, y_pred = pipeline_mlp.predict_and_evaluate(test_loader)
              # ---------- Optuna pruning con F1 ----------
              trial.report(f1, epoch)
              if trial.should_prune():
                  raise optuna.exceptions.TrialPruned()
              # ---------- Early stopping (por loss) ----------
              if avg_val_loss + min_delta < best_val_loss:
                  best_val_loss = avg_val_loss
                  best_model_state = pipeline_mlp.model.state_dict()
                  epochs_no_improve = 0
              else:
                  epochs_no_improve += 1
                  if epochs_no_improve >= patience:
                      print('Se interrumpe ejecucion')
                      break

              # ---------- Guarda el modelo del mejor trial según F1 ----------
              try:
                  if trial.number == 0 or f1 > trial.study.best_value:
                      self.eval_model(
                          test_loader=test_loader,
                          best_model= copy.deepcopy(pipeline_mlp),
                          epoch_number=epoch,
                          best_params=params
                      )

              except ValueError:
                pass

        #-------------Visualization metrics-----------------
        get_metrics(y_test, y_pred)

        return f1

    def eval_model(self, test_loader, best_model, epoch_number, best_params):
        avg_val_loss, f1, y_true, y_pred = best_model.predict_and_evaluate(test_loader)

        self.results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'cm': confusion_matrix(y_true, y_pred),
            'best_params': best_params,
            'model_state_dict': best_model.model.state_dict(),
            'epoch_number': epoch_number
        }

    #----------- Método para obtener los resultados -----------
    def get_results(self):
      return self.results