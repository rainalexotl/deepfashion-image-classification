# DeepFashion Image Classification

## Getting Started
Clone the project structure
```
git clone https://github.com/rainalexotl/deepfashion-image-classification.git 
```

Then get the data! Make sure to [download the following files from the Google Drive link](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html) to `data/raw`.

Category and Attribute Prediction Benchmark
- Img/img.zip
- Anno_fine/train_cate.txt
- Anno_fine/train.txt
- Anno_fine/test_cate.txt
- Anno_fine/test.txt
- Anno_fine/val_cate.txt
- Anno_fine/val.txt
- Anno_fine/list_category_cloth.txt

Unzip img.zip then run the following from root:
```
python3 src/data/organize_dataset.py
```

## Training a model
```
python3 train.py --config config_filename.yaml
```

## Training a new model - Workflow
| What You Want to Do | What to Do |
| :---- | :---- |
| New architecture | Define a new model in src/models/, register it in `factory.py`, create a new config YAML file pointing to it. Make sure to place it into `configs` |
| New hyperparameters (only) | Create a new config YAML (or edit an existing one) with different values (like learning rate, batch size, epochs, etc.). |
| Run experiment | Launch with a simple command: python train.py --config my_new_experiment.yaml |

## Continue training from checkpoint
Create new config (if necessary) and run the following
```
python3 train.py --config new_config_file.yaml --checkpoint path/to/checkpoint.pt

# example
python3 train.py --config baseline_epochs20.yaml --checkpoint experiments/baseline/checkpoints/last_model.pt
```

## Evaluating a model
Make sure that the config file and checkpoint file align. It will most likely be whichever config file was used to train the model at checkpoint.
```
python3 eval.py --config config_filename.yaml --checkpoint path/to/checkpoint.pt [-s]
```
Add `-s` to save evaluation results. This will save the classification report to a json and the predictions (true vs pred) as a csv in `experiments/experiment_name/predictions/`