# Siamese Network on Stylized Flowers.

This project implements a **Siamese Neural Network** using TensorFlow to learn similarity between original and stylized (e.g., Cezanne-style) flower images. Trained with **contrastive loss**, the model distinguishes between matching and non-matching class pairs across different visual domains. Includes training, evaluation, and helper utilities—ideal for research in **domain adaptation** and metric learning.

## 🌸 Datasets
- **Original Domain**: `/data/original`
- **Stylized Domain**: `/data/stylized`

Each domain has:
- 5 categories (flowers)
- Train & Validation folders with identical class structure

## 🧠 Model Architecture
A dual-branch Siamese network using a shared ConvNet. Trained using **Contrastive Loss**.

## 🏋️ Training
- Optimizer: RMSprop
- Epochs: 50
- Batch Size: 32

## 📊 Results
Include accuracy and contrastive loss plot.

## 🚀 Run It
```bash
python train.py
```

## 📁 Project Structure
```
├── train.py
├── evaluate.py
├── utils.py
├── README.md
├── requirements.txt
├── .gitignore
└── data/
    ├── original/
    │   ├── train/
    │   └── validation/
    └── stylized/
        ├── train/
        └── validation/
```
