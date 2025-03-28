# AI Application – Final Assignment

**Project:** Classification of Handwritten Digits with PyTorch

## Project Overview

This project explores the use of Convolutional Neural Networks (CNN) for classifying handwritten digits. It includes data processing, augmentation with custom images, model training, and live predictions.

The main notebook is:  
**`scr/KI_anwendung_ManuelCaipo.ipynb`**

---

## Project Structure (Recommended)

```
.
├── aufgabeki/                          # Main project folder
│   ├── aufgabekai/                     # Python package (recognized by Poetry)
│   │   └── __main__.py                 # Entry point (optional but useful)
│   │
│   └── __init__.py                     # Optional: makes folder a Python module
│
├── scr/                                # Source code and notebooks
│   ├── KI_anwendung_ManuelCaipo.ipynb
│   └── MNIST_Manuel Caipo.ipynb
│
├── utils.py                            # Utility functions (assumed at root)
│
├── AI Application Presentation WS 2024_286577.pptx
├── AI Application Presentation WS 2024_286577_ManuelCaipo.pdf
│
├── .gitignore                          # Git ignore rules
├── poetry.lock                         # Dependency lock file
├── pyproject.toml                      # Poetry project definition
└── README.md                           # Project description and usage

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

This project was carried out as part of the module **"AI Application"**  
at [Hochschule Furtwangen University (HFU)](https://www.hs-furtwangen.de/).  
Author: **Manuel Caipo Ccoa**
