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
python3 train.py --config <filename>.yaml
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
python3 train.py --config <new_config.yaml> --checkpoint <path_to_checkpoint_file.pt>

# example
python3 train.py --config baseline_epochs20.yaml --checkpoint experiments/baseline/checkpoints/last_model.pt
```