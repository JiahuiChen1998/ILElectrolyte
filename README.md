# ILElectrolyte

Prediction models for electrochemical properties of Ionic Liquids

## Overview

ILElectrolyte is a machine learning framework for predicting electrochemical properties of ionic liquids. This repository provides a basic structure for building, training, and deploying molecular property prediction models.

## Features

- **Modular Architecture**: Clean separation of model, data loading, training, and prediction components
- **PyTorch-based**: Built on PyTorch for flexibility and GPU acceleration
- **Easy to Extend**: Simple base architecture that can be customized for specific tasks
- **Comprehensive Testing**: Unit tests for core components
- **Example Scripts**: Ready-to-use examples for training and prediction

## Project Structure

```
ILElectrolyte/
├── src/                      # Source code
│   ├── __init__.py          # Package initialization
│   ├── model.py             # Model architecture definition
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── train.py             # Training logic
│   ├── predict.py           # Prediction/inference utilities
│   └── utils.py             # Utility functions
├── tests/                    # Unit tests
│   ├── test_model.py        # Model tests
│   ├── test_data_loader.py  # Data loader tests
│   └── test_utils.py        # Utility function tests
├── examples/                 # Example scripts
│   ├── train_example.py     # Training example
│   └── predict_example.py   # Prediction example
├── configs/                  # Configuration files
│   └── config.yaml          # Model and training configuration
├── data/                     # Data directory (for datasets)
├── models/                   # Saved model checkpoints
├── requirements.txt          # Python dependencies
├── setup.py                 # Package setup file
└── README.md                # This file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/JiahuiChen1998/ILElectrolyte.git
cd ILElectrolyte
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Training a Model

```python
from src.model import MolecularPropertyPredictor
from src.data_loader import create_data_loader
from src.train import Trainer
import numpy as np

# Prepare your data
train_features = np.random.randn(1000, 100)
train_labels = np.random.randn(1000, 1)

# Create data loader
train_loader = create_data_loader(train_features, train_labels, batch_size=32)

# Initialize model
model = MolecularPropertyPredictor(input_dim=100, hidden_dim=128, output_dim=1)

# Train
trainer = Trainer(model, device='cpu', learning_rate=0.001)
history = trainer.train(train_loader, epochs=50)
```

### Making Predictions

```python
from src.model import MolecularPropertyPredictor
from src.predict import Predictor
import numpy as np

# Initialize and load model
model = MolecularPropertyPredictor(input_dim=100, hidden_dim=128, output_dim=1)
predictor = Predictor.from_checkpoint('models/best_model.pth', model)

# Make predictions
test_features = np.random.randn(10, 100)
predictions = predictor.predict(test_features)
```

### Running Examples

Train a model with the example script:
```bash
python examples/train_example.py
```

Make predictions with a trained model:
```bash
python examples/predict_example.py
```

## Configuration

Edit `configs/config.yaml` to customize model architecture, training parameters, and data paths:

```yaml
model:
  input_dim: 100
  hidden_dim: 128
  output_dim: 1
  dropout: 0.2

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  device: "cpu"
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Run tests with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Model Architecture

The base `MolecularPropertyPredictor` model is a simple feedforward neural network with:
- Input layer
- Two hidden layers with ReLU activation
- Dropout for regularization
- Output layer

This architecture can be extended or replaced based on your specific needs.

## Data Format

The model expects:
- **Features**: NumPy arrays or PyTorch tensors of shape `(n_samples, n_features)`
- **Labels**: NumPy arrays or PyTorch tensors of shape `(n_samples, n_outputs)`

You can customize the `data_loader.py` module to handle your specific data format.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:

```
@software{ilelectrolyte2024,
  title = {ILElectrolyte: Prediction Models for Ionic Liquid Properties},
  author = {ILElectrolyte Team},
  year = {2024},
  url = {https://github.com/JiahuiChen1998/ILElectrolyte}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
