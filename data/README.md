# Data Directory

Place your dataset files here.

## Expected Format

The data loader expects molecular features and property labels. You can implement custom loading logic in `src/data_loader.py`.

## Example Structure

```
data/
├── train.csv          # Training data
├── validation.csv     # Validation data
├── test.csv           # Test data
└── README.md          # This file
```

## Data Preparation

1. Extract molecular features (fingerprints, descriptors, etc.)
2. Prepare labels for target properties
3. Save in a format compatible with your data loader implementation

## Notes

- This directory is included in `.gitignore` to prevent committing large data files
- Consider using separate directories for raw and processed data
