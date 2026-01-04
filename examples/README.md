# Examples Directory

This directory contains example scripts demonstrating how to use the ILElectrolyte framework.

## Available Examples

### 1. Training Example (`train_example.py`)

Demonstrates how to:
- Load and prepare data
- Initialize a model
- Train the model
- Save checkpoints

Run with:
```bash
python examples/train_example.py
```

### 2. Prediction Example (`predict_example.py`)

Demonstrates how to:
- Load a trained model
- Make predictions on new data
- Evaluate model performance
- Save predictions

Run with:
```bash
python examples/predict_example.py
```

## Customization

These examples use dummy data for demonstration. To use with real data:

1. Replace the data generation code with actual data loading
2. Adjust model architecture parameters as needed
3. Modify training hyperparameters in the configuration

## Tips

- Start with small datasets to verify the pipeline works
- Use GPU (`device='cuda'`) for faster training on large datasets
- Monitor training progress to detect overfitting
- Save multiple checkpoints during training for model selection
