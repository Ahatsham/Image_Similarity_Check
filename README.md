# Siamese Network on Stylized Flowers

This project trains a **Siamese Network** to measure visual similarity between original flower images and their stylized versions (e.g., Cezanne drawings). The model learns whether two images represent the same class even across domain shifts.

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
Include accuracy and contrastive loss plots.

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
