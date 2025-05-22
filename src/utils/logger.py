import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

class ExperimentLogger():
    def __init__(self, save_dir, include_acc=False):
        self.save_dir = save_dir
        if include_acc:
            self.history = { 'train_loss': [], 'val_loss': [], 'val_acc': [] }
        else:
            self.history = { 'train_loss': [], 'val_loss': [] }

    def log(self, train_loss, val_loss, val_acc=None):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        if val_acc:
            self.history['val_acc'].append(val_acc)

    def save(self):
        df = pd.DataFrame(self.history)
        csv_path = os.path.join(ROOT, self.save_dir, 'logs', 'history.csv')
        df.to_csv(csv_path, index=False)