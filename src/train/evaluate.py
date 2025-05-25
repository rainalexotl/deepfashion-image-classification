import os
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader
from transformers import ( ViTForImageClassification, ViTImageProcessor,
                          SwinForImageClassification, AutoImageProcessor )
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from src.models.factory import get_model
from src.data.v2.dataset import get_dataloaders, get_datasets

ROOT = Path(__file__).resolve().parents[2]

def evaluate(config, checkpoint_path, save_results=True):
    
    print(f"Evalutating model at checkpoint {checkpoint_path}\n")

    model = get_model(config)
    checkpoint = torch.load(os.path.join(ROOT, checkpoint_path))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loader = get_dataloaders(config, train=False)

    preds = []
    true = []
    model.eval()
    with torch.no_grad():
        for X_test, y_test in tqdm(test_loader):
            batch_preds = model(X_test)
            preds.extend(batch_preds.argmax(dim=1).tolist())
            true.extend(y_test.tolist())

    print(classification_report(true, preds, zero_division=0.0))

    if save_results:
        print("Saving results... ", end='')

        save_dir = os.path.join(config['train']['save_dir'], 'predictions')
        os.makedirs(os.path.join(ROOT, save_dir), exist_ok=True)
        report = classification_report(true, preds, zero_division=0.0, output_dict=True)

        # save report as json
        with open(os.path.join(ROOT, save_dir, 'report.json'), 'w') as f:
            json.dump(report, f, indent=4)

        # save predictions
        df = pd.DataFrame({'true': true, 'preds': preds})
        df.to_csv(os.path.join(ROOT, save_dir, 'predictions.csv'), index=False)

        print("Done.")

def hf_evaluate(config, checkpoint_path, save_results=True):
    print(f"Evalutating HF model at checkpoint {checkpoint_path}\n")

    if config['model'].get('type', '') == 'vit':
        model = ViTForImageClassification.from_pretrained(checkpoint_path)
        processor = ViTImageProcessor.from_pretrained(config['model']['name'])
    elif config['model'].get('type', '') == 'swin': 
        model = SwinForImageClassification.from_pretrained(checkpoint_path)
        processor = AutoImageProcessor.from_pretrained(config['model']['name'])
    else:
        raise ValueError(f"Unsupported model type: {config['model']['type']}")

    test_dataset = get_datasets(config, processor, train=False)
    id2label = {i: label for i, label in enumerate(test_dataset.classes)}
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluating on {device}")

    model.to(device)
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Validating..."):
            X_val = batch['pixel_values'].to(device)
            y_val = batch['labels'].to(device)

            outputs = model(pixel_values=X_val)
            preds = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(y_val.cpu().tolist())

    label_true_named = [id2label.get(i, str(i)) for i in all_true]
    label_pred_named = [id2label.get(i, str(i)) for i in all_preds]
    print(classification_report(label_true_named, label_pred_named))

    if save_results:
        print("Saving results... ", end='')

        save_dir = os.path.join(config['experiment']['save_root'], config['experiment']['name'], 'predictions')
        os.makedirs(os.path.join(ROOT, save_dir), exist_ok=True)
        report = classification_report(label_true_named, label_pred_named, zero_division=0.0, output_dict=True)

        # save report as json
        with open(os.path.join(ROOT, save_dir, 'report.json'), 'w') as f:
            json.dump(report, f, indent=4)

        # save predictions
        df = pd.DataFrame({'true': all_true, 'preds': all_preds})
        df.to_csv(os.path.join(ROOT, save_dir, 'predictions.csv'), index=False)

        # save confusion matrix
        save_dir = os.path.join(config['experiment']['save_root'], config['experiment']['name'], 'plots')
        os.makedirs(os.path.join(ROOT, save_dir), exist_ok=True)
        
        plt.figure(figsize=(15, 15))
        tick_labels = list(id2label.values())
        cm = confusion_matrix(label_true_named, label_pred_named)
        conf_mat = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                               xticklabels=tick_labels, yticklabels=tick_labels)
        plt.yticks(rotation=0) 
        plt.xticks(rotation=90) 
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        fig = conf_mat.get_figure()
        fig.savefig(os.path.join(ROOT, save_dir, 'confusion_matrix.png')) 
        plt.close()

        print("Done.")
