# Glaucoma Detection - SegFormer Based Semantic Segmentation

## Project Structure

```
glaucoma_detection/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── prepare_dataset.py       # Dataset loading & preprocessing
│
├── model/
│   └── segformer_model.py       # SegFormer model definition
│
├── train.py                     # Training script
├── evaluate.py                  # Evaluation & metrics
├── predict.py                   # Inference & CDR computation
└── utils.py                     # Augmentation, helpers
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
