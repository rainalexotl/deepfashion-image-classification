import os
from pathlib import Path
import shutil
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from transformers import ( ViTForImageClassification, ViTImageProcessor,
                            SwinForImageClassification, AutoImageProcessor,
                            Trainer, TrainingArguments, EarlyStoppingCallback )
import evaluate

from src.data.v2.dataset import get_datasets, get_dataloaders
from src.models.factory import get_model
from src.utils.logger import ExperimentLogger

ROOT = Path(__file__).resolve().parents[2]

class BaseTrainer():
    def __init__(self, config, checkpoint_path=None):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print("Training on CUDA!")
        self.save_dir = os.path.join(ROOT, config['experiment']['save_root'], config['experiment']['name'])
        self.logger = ExperimentLogger(self.save_dir)

        os.makedirs(os.path.join(ROOT, self.save_dir), exist_ok=True)
        os.makedirs(os.path.join(ROOT, self.save_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(ROOT, self.save_dir, 'logs'), exist_ok=True)
        shutil.copy(
            os.path.join(ROOT, 'configs', f"{config['model']['experiment_name']}.yaml"), 
            os.path.join(ROOT, self.save_dir, 'config.yaml')
        )

        self.model = get_model(config).to(self.device)
        self.train_loader, self.val_loader = get_dataloaders(config)

        model_params =  [p for p in self.model.parameters() if p.requires_grad]
        self.criterion = nn.CrossEntropyLoss()
        weight_decay = config['train'].get('weight_decay', 0)
        self.optimizer = torch.optim.Adam(model_params, lr=config['train']['lr'], 
                                          weight_decay=weight_decay)

        # for checkpointing
        self.start_epoch = 0
        self.curr_epoch = self.start_epoch # for saving in case of keyboard interrupt
        self.best_val_loss = float('inf')
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def train(self):
        for e in range(self.start_epoch, self.config['train']['epochs']): # start_epoch in case of checkpoints
            # print(f"Epoch {e:3}... ", end='', flush=True)
            self.curr_epoch = e

            train_epoch_loss = 0
            val_epoch_loss = 0

            for X_train, y_train in tqdm(self.train_loader, desc=f'Training epoch {e}', leave=False):
                # X_train, y_train = X_train.to(self.device), y_train.to(self.device)
                train_preds = self.model(X_train)
                loss_train = self.criterion(train_preds, y_train)
                train_epoch_loss += loss_train.item()

                self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()

            with torch.no_grad():
                for X_val, y_val in tqdm(self.val_loader, desc=f'Validation epoch {e}', leave=False):
                    # X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    val_preds = self.model(X_val)
                    loss_val = self.criterion(val_preds, y_val)
                    val_epoch_loss += loss_val.item()

            train_epoch_loss /= len(self.train_loader)
            val_epoch_loss /= len(self.val_loader)
            self.logger.log(train_epoch_loss, val_epoch_loss)
            print(f"Epoch {e:3} Train loss: {train_epoch_loss}\tVal loss: {val_epoch_loss}")

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

    def _resolve_model_param_group(self, obj, target_params):
        for attr in target_params.split('.'):
            obj = getattr(obj, attr) if not attr.isdigit() else obj[int(attr)]
        return obj


    def _load_checkpoint(self, path):
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.history = checkpoint['history']
        self.best_val_loss = min(checkpoint['history']['val_loss'])

    def save_checkpoint(self, path):
        torch.save({
            'epoch': self.curr_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.logger.history
        }, path)

class HFTrainer():
    def __init__(self, config, checkpoint_path=None):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.save_dir = os.path.join(ROOT, config['experiment']['save_root'], config['experiment']['name'])
        self.experiment_name = config['experiment']['name']
        self.logger = ExperimentLogger(self.save_dir, include_acc=True)
        os.makedirs(os.path.join(ROOT, self.save_dir), exist_ok=True)
        os.makedirs(os.path.join(ROOT, self.save_dir, 'logs'), exist_ok=True)
        shutil.copy(
            os.path.join(ROOT, 'configs', f"{config['experiment']['name']}.yaml"), 
            os.path.join(ROOT, self.save_dir, 'config.yaml')
        )

        if config['model']['type'] == 'vit':
            self.processor = ViTImageProcessor.from_pretrained(config['model']['name'])
            self.model = ViTForImageClassification.from_pretrained(
                config['model']['name'], num_labels=config['model']['num_classes'],
                ignore_mismatched_sizes=True
            )
        elif config['model']['type'] == 'swin':
            self.processor = AutoImageProcessor.from_pretrained(config['model']['name'])
            self.model = SwinForImageClassification.from_pretrained(
                config['model']['name'], num_labels=config['model']['num_classes'],
                ignore_mismatched_sizes=True
            )
        else:
            raise ValueError("Missing model 'type' in config file.")
        
        self.train_dataset, self.val_dataset = get_datasets(config, self.processor, train=True)
    
        self.training_args = TrainingArguments(
            output_dir=self.save_dir,
            overwrite_output_dir=True,
            eval_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2, # save last
            learning_rate=config['train']['learning_rate'],
            per_device_train_batch_size=config['train']['batch_size'],
            per_device_eval_batch_size=config['train']['batch_size'],
            # This will pick up from the last checkpoint (e.g., epoch 3) and train up to the desired number of epochs
            num_train_epochs=config['train']['epochs'],
            logging_strategy='epoch',
            load_best_model_at_end=True, # save best
            metric_for_best_model='loss'
        )

    def _compute_acc(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        accuracy = evaluate.load('accuracy')
        acc = accuracy.compute(predictions=predictions, references=labels)
        return acc
    
    def train(self):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self._compute_acc,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config['train']['patience'])]
        )
        trainer.train(resume_from_checkpoint=self.checkpoint_path)

        # TODO: handling continuing from checkpoint?
        log_history = trainer.state.log_history
        for i in range(0, len(log_history) - 1, 2):
            self.logger.log(log_history[i]['loss'], 
                            log_history[i+1]['eval_loss'], 
                            log_history[i+1]['eval_accuracy'])
            
        self.logger.save()
        best_model = trainer.state.best_model_checkpoint
        print(f"Best checkpoint was saved at: {trainer.state.best_model_checkpoint}")
        if self.config['model'].get('type', '') == 'vit':
            model = ViTForImageClassification.from_pretrained(best_model)
        elif self.config['model'].get('type', '') == 'swin':
            model = SwinForImageClassification.from_pretrained(best_model)
        else:
            raise ValueError("Missing model 'type' in config file. Should be swin or vit")
        torch.save(model.state_dict(), os.path.join(self.save_dir, f"{self.config['experiment']['name']}_best_model.pt"))

class TimmTrainer():
    def __init__():
        pass