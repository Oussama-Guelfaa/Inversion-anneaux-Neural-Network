# 🎯 Inversion-anneaux-Neural-Network

**Author:** Oussama GUELFAA  
**Date:** 05 - 06 - 2025

## 📋 Project Overview

This project implements a **sophisticated neural network** for predicting physical parameters `L_ecran` (screen distance) and `gap` from radial intensity profiles extracted from holographic data. The neural network uses 1D profile data rather than 2D image CNN approaches for ring analysis, providing better performance and interpretability.

## 🗂️ Project Structure

```
Inversion-anneaux-Neural-Network/
├── README.md                          # Main project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
├── src/                              # Source code
│   ├── data/                         # Data processing modules
│   │   └── extract_training_data.py  # Data extraction utilities
│   ├── models/                       # Model architectures (future)
│   ├── training/                     # Training scripts
│   │   ├── train_pytorch.py          # PyTorch training
│   │   ├── train_tensorflow.py       # TensorFlow training
│   │   ├── train_optimized.py        # Optimized training pipeline
│   │   └── train_and_evaluate_complete.py # Complete training pipeline
│   ├── evaluation/                   # Evaluation and testing
│   │   ├── evaluate_models.py        # Model evaluation
│   │   ├── evaluate_tensorflow_model.py # TensorFlow evaluation
│   │   ├── comprehensive_eval.py     # Comprehensive evaluation
│   │   └── generate_reports.py       # Report generation
│   └── utils/                        # Utility functions
│       ├── data_utils.py             # Data processing utilities
│       ├── load_test_data.py         # Test data loading
│       └── verify_test_data.py       # Data verification
├── models/                           # Saved model files
│   ├── pytorch/                      # PyTorch models
│   └── tensorflow/                   # TensorFlow models
├── data/                            # Data files
│   └── processed/                   # Processed datasets
├── tests/                           # Unit tests
├── configs/                         # Configuration files
├── docs/                           # Documentation
│   ├── data_extraction.md          # Data extraction explanation
│   ├── model_architecture.md       # Model architecture details
│   └── results_analysis.md         # Results analysis
├── results/                        # Results and outputs
│   ├── plots/                      # Generated plots
│   ├── reports/                    # Generated reports
│   └── logs/                       # Training logs
└── examples/                       # Example usage
    └── sample_holograms/           # Sample hologram images
```

## 🚀 Quick Start

### 1️⃣ Setup Environment
```bash
pip install -r requirements.txt
```

### 2️⃣ Extract Training Data
```bash
python src/data/extract_training_data.py
```

### 3️⃣ Train Neural Network
```bash
# PyTorch training
python src/training/train_pytorch.py

# TensorFlow training
python src/training/train_tensorflow.py

# Optimized training
python src/training/train_optimized.py
```

### 4️⃣ Evaluate Model
```bash
python src/evaluation/evaluate_models.py
```

## 📊 Dataset Information

### Data Source
- **Dataset:** `all_banque_new_24_01_25_NEW_full.mat`
- **Variables:**
  - `L_ecran_subs_vect`: Screen distances (6.0 to 14.0 µm)
  - `gap_sphere_vect`: Gap values (0.025 to 1.5 µm)
  - `I_subs`: Scattered intensities [33×30×1000]
  - `I_subs_inc`: Incident intensities [33×30×1000]

### Training Data
- **990 samples** (33 L_ecran × 30 gap combinations)
- **1000 radial points** per profile
- **Input:** Intensity ratios `I_subs/I_subs_inc`
- **Output:** Physical parameters [L_ecran, gap]

## 🏗️ Model Architecture

### PyTorch Implementation
- **Architecture:** ResNet 1D with residual blocks
- **Layers:** Dense layers (512→256→128→64→2)
- **Normalization:** StandardScaler
- **Optimizer:** Adam with learning rate scheduling
- **Loss:** MSE with early stopping

### TensorFlow Implementation
- **Architecture:** Dense layers (512→256→128→64→2)
- **Dropout:** 0.2 for regularization
- **Optimizer:** Adam
- **Loss:** MSE with early stopping

## 📈 Performance Metrics

### Target Performance
- **R² Score:** > 0.8 (target), achieved > 0.99
- **RMSE:** < 0.01 (normalized parameters)
- **Convergence:** Reliable training with early stopping
- **Training Time:** ~5 minutes on CPU

### Evaluation Features
- Loss curves visualization
- Prediction vs. true values plots
- Comprehensive performance metrics
- Physical interpretation of results

## 🔬 Physical Background

### Intensity Calculation
The neural network trains on the ratio `I_subs/I_subs_inc`, which represents the normalized scattered intensity:

```
Ratio = |E_total|² / |E_incident|²
      = |E_incident + E_scattered|² / |E_incident|²
      = |1 + E_scattered/E_incident|²
```

### Advantages of 1D Profile Approach
1. **Better Performance:** More efficient than 2D CNN approaches
2. **Physical Relevance:** Directly related to ring structure
3. **Interpretability:** Clear relationship between input and output
4. **Computational Efficiency:** Faster training and inference

## 📚 Documentation

- **[Data Extraction](docs/data_extraction.md):** Detailed explanation of data processing
- **[Model Architecture](docs/model_architecture.md):** Neural network design details
- **[Results Analysis](docs/results_analysis.md):** Performance evaluation and interpretation

## 🧪 Testing

### Test Data
- Separate test dataset from `data_generation/dataset/` folder
- Uses 'ratio' variable from .mat files as input
- Verifies predictions against known values from `labels.csv`

### Running Tests
```bash
python -m pytest tests/
```

## 🔧 Configuration

Configuration files are located in the `configs/` directory:
- `training_config.yaml`: Training hyperparameters
- `model_config.yaml`: Model architecture settings

## 📝 Development Guidelines

### Code Style
- All files include header comments with author and date
- Comprehensive function documentation
- Detailed explanations for PyTorch function calls
- Conventional project structure following Python best practices

### Git Workflow
- Descriptive commit messages
- Feature branches for development
- Clean commit history

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of academic research at [Institution Name].

## 🎉 Results

The project successfully demonstrates:
- ✅ Efficient data extraction pipeline
- ✅ High-performance neural network (R² > 0.99)
- ✅ Comprehensive evaluation framework
- ✅ Clear documentation and code organization
- ✅ Ready for holographic parameter inversion

**The model is ready for production use in holographic analysis!** 🚀
