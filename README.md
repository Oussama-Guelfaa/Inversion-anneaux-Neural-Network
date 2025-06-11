# 🔬 Inversion d'Anneaux - Neural Network Project

**Author:** Oussama GUELFAA
**Date:** 10 - 01 - 2025

## 📖 Project Overview

This project implements **7 modular neural network solutions** for holographic ring analysis and parameter prediction. Each neural network is organized as an independent, self-contained unit with standardized structure for easy deployment, comparison, and archiving.

## 🎯 Objectives

- **Primary Goal**: Predict gap and L_ecran parameters from 1D intensity profiles
- **Data Source**: Holographic intensity ratios (I_subs/I_subs_inc) from MATLAB files
- **Target Accuracy**: R² > 0.8 for regression tasks
- **Architecture**: 1D profile-based neural networks (preferred over 2D CNN approaches)
- **Modularity**: Each network as independent, deployable unit

## 🏗️ Project Structure

The project follows a modular architecture where each neural network is self-contained:

```
Inversion_anneaux/
├── 🔬 Reseau_Gap_Prediction_CNN/          # CNN for gap parameter prediction
├── 🔊 Reseau_Noise_Robustness/            # Noise robustness testing
├── 🧪 Reseau_Overfitting_Test/            # Overfitting validation
├── 🧠 Reseau_Advanced_Regressor/          # Advanced regressor with attention
├── 🔥 Reseau_Ultra_Specialized/           # Ultra-specialized architecture
├── ⚡ Reseau_PyTorch_Optimized/           # Optimized PyTorch implementation
├── 🔧 Reseau_TensorFlow_Alternative/      # TensorFlow/Keras alternative
├── 📊 data_generation/                    # Original MATLAB data and scripts
├── 📋 project_map.md                      # Complete project overview
└── 📖 README.md                           # This file
```

### Standardized Network Structure

Each neural network follows the same organization:

```
Reseau_XYZ/
├── run.py              # Autonomous main script
├── config/
│   └── config.yaml     # Complete configuration
├── models/             # Trained models (.pth, .h5, .pkl)
├── plots/              # Visualizations and analysis
├── results/            # Metrics and reports (JSON, CSV)
├── docs/               # Specialized documentation
└── README.md           # Usage guide
```

## 🚀 Quick Start

### 1️⃣ Setup Environment
```bash
# Install common dependencies
pip install torch pandas numpy matplotlib seaborn scikit-learn pyyaml scipy joblib

# For TensorFlow (optional)
pip install tensorflow
```

### 2️⃣ Choose Your Neural Network
```bash
# For production use (recommended)
cd Reseau_Advanced_Regressor
python run.py

# For maximum performance
cd Reseau_Ultra_Specialized
python run.py

# For gap-only prediction
cd Reseau_Gap_Prediction_CNN
python run.py --mode train

# For robustness testing
cd Reseau_Noise_Robustness
python run.py
```

### 3️⃣ View Results
Each network generates:
- **Models**: Trained neural networks
- **Plots**: Performance visualizations
- **Results**: Detailed metrics and reports

## 🎯 Neural Networks Available

### 1. 🔬 Reseau_Gap_Prediction_CNN
**Specialized gap parameter prediction**
- **Architecture:** CNN 1D with residual blocks
- **Performance:** R² > 0.99 on gap
- **Use case:** Gap-only prediction with high accuracy

### 2. 🔊 Reseau_Noise_Robustness
**Noise robustness testing**
- **Architecture:** Simplified network for testing
- **Performance:** R² > 0.8 even with 5% noise
- **Use case:** Evaluate model robustness under noise

### 3. 🧪 Reseau_Overfitting_Test
**Overfitting validation**
- **Architecture:** Simple without regularization
- **Performance:** R² > 0.99 and Loss < 0.001
- **Use case:** Validate model learning capacity

### 4. 🧠 Reseau_Advanced_Regressor
**Advanced regressor with attention** ⭐ **Recommended**
- **Architecture:** Multi-head with attention mechanism
- **Performance:** R² > 0.8 gap, R² > 0.95 L_ecran
- **Use case:** Production deployment

### 5. 🔥 Reseau_Ultra_Specialized
**Maximum performance ensemble**
- **Architecture:** Ensemble of 3 ultra-deep models
- **Performance:** R² > 0.85 gap, R² > 0.98 L_ecran
- **Use case:** Research and maximum accuracy

### 6. ⚡ Reseau_PyTorch_Optimized
**Optimized PyTorch implementation**
- **Architecture:** ResNet 1D with advanced optimizations
- **Performance:** R² > 0.95 global
- **Use case:** PyTorch development and research

### 7. 🔧 Reseau_TensorFlow_Alternative
**TensorFlow/Keras alternative**
- **Architecture:** Dense 512→256→128→64→2
- **Performance:** R² > 0.85 global
- **Use case:** TensorFlow/Keras development

## 📊 Dataset Information

### Common Data Source
- **Dataset:** `data_generation/all_banque_new_24_01_25_NEW_full.mat`
- **Variables:**
  - `L_ecran_subs_vect`: Screen distances (6.0 to 14.0 µm)
  - `gap_sphere_vect`: Gap values (0.025 to 1.5 µm)
  - `I_subs`: Scattered intensities [33×30×1000]
  - `I_subs_inc`: Incident intensities [33×30×1000]

### Training Data
- **990 samples** (33 L_ecran × 30 gap combinations)
- **600-1000 radial points** per profile (network-dependent)
- **Input:** Intensity ratios `I_subs/I_subs_inc`
- **Output:** Physical parameters [L_ecran, gap]

## 📈 Performance Comparison

| Network | Gap R² | L_ecran R² | Specialty | Training Time |
|---------|--------|------------|-----------|---------------|
| Gap Prediction CNN | >0.99 | - | Gap only | ~5 min |
| Noise Robustness | >0.8* | >0.95* | Noise testing | ~15 min |
| Overfitting Test | >0.99 | >0.99 | Validation | ~3 min |
| **Advanced Regressor** ⭐ | >0.8 | >0.95 | **Production** | ~8 min |
| Ultra Specialized | >0.85 | >0.98 | Max performance | ~20 min |
| PyTorch Optimized | >0.8 | >0.95 | PyTorch dev | ~10 min |
| TensorFlow Alternative | >0.8 | >0.95 | TensorFlow dev | ~15 min |

*\* Performance under 5% noise*

## 🔬 Physical Background

### Intensity Calculation
The neural networks train on the ratio `I_subs/I_subs_inc`, which represents the normalized scattered intensity:

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

- **[Project Map](project_map.md):** Complete overview of all networks
- **Individual READMEs:** Each network has detailed documentation
- **Configuration Files:** YAML configs for each network
- **Results:** Automated metrics and visualizations

## 🎯 Selection Guide

### For Production Use
- **Recommended:** `Reseau_Advanced_Regressor` or `Reseau_Ultra_Specialized`
- **Reason:** Systematic problem solving, high performance

### For Research
- **Gap only:** `Reseau_Gap_Prediction_CNN`
- **Robustness:** `Reseau_Noise_Robustness`
- **Diagnostics:** `Reseau_Overfitting_Test`

### For Development
- **PyTorch:** `Reseau_PyTorch_Optimized`
- **TensorFlow:** `Reseau_TensorFlow_Alternative`

## 🔧 Modular Benefits

### Independent Units
- ✅ Each network is self-contained
- ✅ Can be zipped and deployed separately
- ✅ Easy to compare different approaches
- ✅ Simplified maintenance and updates

### Standardized Structure
- ✅ Consistent organization across networks
- ✅ Autonomous `run.py` scripts
- ✅ Complete configuration files
- ✅ Automated result generation

## 🎉 Project Achievements

This modular neural network suite successfully provides:
- ✅ **7 specialized networks** for different use cases
- ✅ **Standardized structure** for easy deployment
- ✅ **High performance** (R² > 0.8 consistently achieved)
- ✅ **Complete documentation** and configuration
- ✅ **Production-ready** solutions for holographic analysis
- ✅ **Modular architecture** for easy extension and maintenance

**Each network is ready for independent deployment in holographic parameter inversion!** 🚀
