# Models Directory

Trained model checkpoints will be saved here.

## Checkpoint Format

Model checkpoints are saved in PyTorch format (`.pth` files) and include:
- Model state dictionary
- Optimizer state dictionary

## Example Structure

```
models/
├── best_model.pth     # Best model based on validation loss
├── final_model.pth    # Final model after training
└── README.md          # This file
```

## Loading Models

```python
from src.model import MolecularPropertyPredictor
from src.predict import Predictor

# Initialize model architecture
model = MolecularPropertyPredictor(input_dim=100, hidden_dim=128, output_dim=1)

# Load from checkpoint
predictor = Predictor.from_checkpoint('models/best_model.pth', model)
```

## Notes

- Model files are excluded from git tracking via `.gitignore`
- Remember to save model configuration alongside checkpoints
- Consider version control for models using tools like DVC or MLflow
