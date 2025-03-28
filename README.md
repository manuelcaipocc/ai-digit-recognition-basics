# AI Application – Final Assignment

**Author:** Manuel Caipo  
**Project:** Classification of Handwritten Digits with PyTorch

## Project Overview

This project explores the use of Convolutional Neural Networks (CNN) for classifying handwritten digits. It includes data processing, augmentation with custom images, model training, and live predictions.

The main notebook is:  
**`scr/notebooks/KI_anwendung_ManuelCaipo.ipynb`**

---

## Project Structure (Recommended)

```
.
├── .git/
├── aufgabeki/                        # Supplementary materials
│
├── scr/                              # Main code
│   ├── notebooks/                    # Jupyter Notebooks
│   │   ├── KI_anwendung_ManuelCaipo.ipynb
│   │   └── MNIST_Manuel Caipo.ipynb
│   │
│   ├── data/                         # Datasets and custom images
│   │   ├── C64_Ziffern_Daten.pkl
│   │   ├── train.zip
│   │   ├── test.zip
│   │   └── images/
│   └── utils.py                      # Utility functions
│
├── presentation/                     # Presentations
│   ├── AI Application Presentation WS 2024_286577.pptx
│   └── AI Application Presentation WS 2024_286577_ManuelCaipo.pdf
│
├── .gitignore
├── poetry.lock
├── pyproject.toml
└── README.md
```

---

## Setup with Poetry and Pyenv

### Requirements

- Python 3.10.x (managed with `pyenv`)
- Dependency management with `Poetry`

### Setup

```bash
# Set Python version
pyenv install 3.10.11
pyenv local 3.10.11

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

### Launch Notebook

```bash
poetry shell
jupyter notebook scr/notebooks/KI_anwendung_ManuelCaipo.ipynb
```

---

## Usage

Ensure the following files are available:

- `scr/utils.py`
- `scr/data/C64_Ziffern_Daten.pkl`
- `scr/data/images/` with your own images

---

## Features

- Preprocessing and transformation of image data
- CNN model architecture and training
- Augmentation with custom images and retraining
- Evaluation and visualization
- Live predictions using camera

---

## Technologies

- **PyTorch**
- **Torchvision**
- **OpenCV**, **NumPy**, **Matplotlib**
- **Torchviz**, **TensorBoard**
- **Poetry**, **Pyenv**
- **Jupyter Notebook**

---

## Author

This project was carried out as part of the module **"AI Application"**.  
Author: **Manuel Caipo**