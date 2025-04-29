import os
from pathlib import Path
import shutil
import pandas as pd

import torch
import torch.nn as nn

from src.data.dataset import get_dataloaders
from src.models.factory import get_model
from src.utils.logger import ExperimentLogger

ROOT = Path(__file__).resolve().parents[2]

class Trainer():
    def __init__(self, config, checkpoint_path=None):
        self.config = config
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_dir = os.path.join(ROOT, config['train']['save_dir'])
        self.logger = ExperimentLogger(self.save_dir)

        os.makedirs(os.path.join(ROOT, self.save_dir), exist_ok=True)
        os.makedirs(os.path.join(ROOT, self.save_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(ROOT, self.save_dir, 'logs'), exist_ok=True)
        shutil.copy(
            os.path.join(ROOT, 'configs', f"{config['model']['name']}.yaml"), 
            os.path.join(ROOT, self.save_dir, 'config.yaml')
        )

        # self.model = get_model(config).to(self.device)
        self.model = get_model(config)
        self.train_loader, self.val_loader = get_dataloaders(config)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['train']['lr'])

        self.start_epoch = 0 # for checkpointing
        self.curr_epoch = self.start_epoch # for saving in case of keyboard interrupt
        self.best_val_loss = float('inf')
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def train(self):
        for e in range(self.start_epoch, self.config['train']['epochs']): # start_epoch in case of checkpoints
            print(f"Epoch {e:3}... ", end='', flush=True)
            self.curr_epoch = e

            train_epoch_loss = 0
            val_epoch_loss = 0

            for X_train, y_train in self.train_loader:
                # X_train, y_train = X_train.to(self.device), y_train.to(self.device)
                train_preds = self.model(X_train)
                loss_train = self.criterion(train_preds, y_train)
                train_epoch_loss += loss_train.item()

                self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()

            with torch.no_grad():
                for X_val, y_val in self.val_loader:
                    # X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    val_preds = self.model(X_val)
                    loss_val = self.criterion(val_preds, y_val)
                    val_epoch_loss += loss_val.item()

            train_epoch_loss /= len(self.train_loader)
            val_epoch_loss /= len(self.train_loader)
            self.logger.log(train_epoch_loss, val_epoch_loss)
            print(f"Train loss: {train_epoch_loss}\tVal loss: {val_epoch_loss}")

            if val_epoch_loss < self.best_val_loss:
                print(f"Saving new best model at epoch {e}")
                self.save_checkpoint(
                    os.path.join(ROOT, self.save_dir, 'checkpoints', 'best_model.pt')
                )
                self.best_val_loss = val_epoch_loss

        # save last checkpoint
        self.save_checkpoint(
            os.path.join(ROOT, self.save_dir, 'checkpoints', 'last_model.pt')
        )
        
        self.logger.save()

    def _load_checkpoint(self, path):
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.history = checkpoint['history']

    def save_checkpoint(self, path):
        torch.save({
            'epoch': self.curr_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.logger.history
        }, path)