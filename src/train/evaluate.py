import os
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd

import torch
from sklearn.metrics import classification_report

from src.models.factory import get_model
from src.data.v2.dataset import get_dataloaders

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