# Glaucoma Detection - SegFormer Based Semantic Segmentation

## Project Structure

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

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

* Download the **REFUGE2 dataset** from: https://refuge.grand-challenge.org/
* Place images in `data/images/` and masks in `data/masks/`.

## Training

```bash
python train.py --epochs 50 --batch_size 8 --lr 6e-5
```

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

## Predict

```bash
python predict.py
```

## Plot Training Results

```bash
python plot_graphs.py
```

## View Preprocessing

```bash
python show_preprocessing.py
```

## Run Application

```bash
python app.py
```
