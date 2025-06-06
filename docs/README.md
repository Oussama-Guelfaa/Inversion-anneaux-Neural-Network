# 📚 Documentation

This directory contains comprehensive documentation for the Inversion-anneaux-Neural-Network project.

## 📄 Documentation Files

### [data_extraction.md](data_extraction.md)
Detailed explanation of the data extraction process from MATLAB files:
- Data structure analysis
- Intensity ratio calculations
- Physical interpretation of the data
- Processing pipeline

### [model_architecture.md](model_architecture.md)
Neural network architecture details:
- PyTorch implementation
- TensorFlow implementation
- Model design rationale
- Performance optimization strategies

### [results_analysis.md](results_analysis.md)
Comprehensive analysis of model performance:
- Training results
- Evaluation metrics
- Performance comparisons
- Physical interpretation of predictions

## 🎯 Key Concepts

### Data Processing
The project processes holographic data by:
1. Extracting intensity ratios `I_subs/I_subs_inc`
2. Converting 2D hologram data to 1D radial profiles
3. Normalizing data for neural network training

### Model Design
The neural networks are designed to:
- Process 1D intensity profiles (1000 points)
- Predict physical parameters [L_ecran, gap]
- Achieve high accuracy (R² > 0.99)
- Provide reliable convergence

### Physical Background
The project focuses on:
- Holographic parameter inversion
- Ring pattern analysis
- Scattering intensity modeling
- Physical parameter prediction

## 📖 Reading Order

For new users, we recommend reading the documentation in this order:
1. **data_extraction.md** - Understand the data processing
2. **model_architecture.md** - Learn about the neural network design
3. **results_analysis.md** - Review performance and results

## 🔗 Related Files

- Main project README: `../README.md`
- Source code documentation: `../src/`
- Example usage: `../examples/`
- Configuration files: `../configs/`
