# Glaucoma Detection - SegFormer Based Semantic Segmentation

## Project Structure

```
## 📂 Project Structure

```text
Glaucoma-detection-using-Transformer/
│
├── data/
│   └── prepare_dataset.py
│
├── model/
│
├── predictions/
│
├── results/
│
├── train.py
├── run_train.py
├── evaluate.py
├── predict.py
├── utils.py
├── show_preprocessing.py
├── plot_graphs.py
├── app.py
│
├── requirements.txt
└── README.md
```

```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset
- Download REFUGE2 dataset from: https://refuge.grand-challenge.org/
- Place images in `data/images/` and masks in `data/masks/`

## Training
```bash
python train.py --epochs 50 --batch_size 8 --lr 6e-5
```

## Evaluation
```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

## Inference
```bash
python predict.py --image path/to/fundus_image.jpg
```
