# DeepFashion Image Classification

## Getting Started
Make sure to [download the following files from the Google Drive link](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html) to `data/raw`.

Category and Attribute Prediction Benchmark
- Img/img.zip
- Anno_fine/train_cate.txt
- Anno_fine/train.txt
- Anno_fine/test_cate.txt
- Anno_fine/test.txt
- Anno_fine/val_cate.txt
- Anno_fine/val.txt
- Anno_fine/list_category_cloth.txt

Then run the following from root:
```
python3 src/data/organize_dataset.py
```