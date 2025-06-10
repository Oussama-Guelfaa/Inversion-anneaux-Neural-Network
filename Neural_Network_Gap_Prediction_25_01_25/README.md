# Gap Parameter Prediction Neural Network

**Author:** Oussama GUELFAA  
**Date:** 25 - 01 - 2025

## Overview

This project implements a sophisticated 1D Convolutional Neural Network (CNN) with residual blocks for predicting gap parameters from holographic intensity profiles. The neural network is specifically designed to handle 1D intensity profile data and achieve high accuracy in gap parameter regression tasks.

## Project Structure

```
Neural_Network_Gap_Prediction_25_01_25/
├── gap_prediction_neural_network.py    # Main implementation
├── models/                              # Saved models and scalers
│   ├── best_model.pth                  # Best model weights
│   ├── final_model.pth                 # Final model weights
│   └── scaler.pkl                      # Data normalization scaler
├── plots/                              # Generated visualizations
│   ├── training_history.png           # Training curves
│   └── evaluation_results.png         # Evaluation plots
├── results/                            # Training and evaluation results
│   ├── training_history.json          # Training metrics history
│   ├── evaluation_results.json        # Evaluation metrics
│   └── model_summary.txt              # Model architecture summary
├── data/                               # Data processing workspace
└── README.md                           # This documentation
```

## Data Analysis

### Dataset Characteristics
- **Input Data**: 990 intensity profiles × 1000 features each
- **Target Parameter**: Gap values ranging from 0.025 to 1.5 µm
- **Data Distribution**: 30 unique gap values with multiple L_ecran configurations
- **Task Type**: Regression (continuous gap parameter prediction)

### Data Preprocessing
1. **Normalization**: StandardScaler applied to intensity profiles for zero mean and unit variance
2. **Train/Validation Split**: 80% training, 20% validation
3. **Data Loading**: Custom PyTorch Dataset class for efficient batch processing
4. **No Data Augmentation**: Clean, synthetic data doesn't require augmentation

## Neural Network Architecture

### Design Rationale

The architecture is specifically designed for 1D intensity profile analysis:

1. **1D Convolutional Layers**: Ideal for capturing local patterns in intensity profiles
2. **Residual Blocks**: Enable deeper networks while maintaining gradient flow
3. **Progressive Channel Increase**: 64 → 128 → 256 → 512 channels for hierarchical feature learning
4. **Global Average Pooling**: Reduces overfitting compared to fully connected layers
5. **Dropout Regularization**: Prevents overfitting with rates of 0.3 and 0.2

### Detailed Architecture

```
Input: (1000,) - 1D intensity profile
├── Conv1D(1→64, kernel=7, stride=2) + BatchNorm + ReLU
├── Conv1D(64→128, kernel=5, stride=2) + BatchNorm + ReLU  
├── ResidualBlock(128→128, kernel=3)
├── Conv1D(128→256, kernel=3, stride=2) + BatchNorm + ReLU
├── ResidualBlock(256→256, kernel=3)
├── Conv1D(256→512, kernel=3, stride=2) + BatchNorm + ReLU
├── GlobalAveragePooling1D()
├── Dense(512→256) + Dropout(0.3) + ReLU
├── Dense(256→128) + Dropout(0.2) + ReLU
└── Dense(128→1) - Gap prediction
```

**Total Parameters**: ~1.2M trainable parameters

### Residual Block Design

Each residual block contains:
- Two Conv1D layers with BatchNorm
- ReLU activations
- Skip connection for gradient flow
- Automatic dimension matching when needed

## Training Methodology

### Hyperparameters
- **Optimizer**: Adam with weight decay (1e-4)
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduler
- **Batch Size**: 32
- **Loss Function**: Mean Squared Error (MSE)
- **Max Epochs**: 150 with early stopping (patience=20)
- **Device**: Automatic GPU/CPU detection

### Training Features
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Learning Rate Scheduling**: Reduces LR when validation loss plateaus
- **Best Model Saving**: Automatically saves the best performing model
- **Comprehensive Monitoring**: Tracks multiple metrics during training

## Evaluation Metrics

### Primary Metrics
1. **R² Score**: Coefficient of determination (target > 0.8)
2. **Tolerance-based Accuracy**: Percentage of predictions within ±0.01 µm
3. **Mean Absolute Error (MAE)**: Average absolute prediction error
4. **Root Mean Squared Error (RMSE)**: Square root of mean squared error

### Tolerance-based Evaluation
The model considers a prediction "correct" if:
```
|predicted_gap - true_gap| ≤ 0.01 µm
```

Additional tolerance levels (±0.005, ±0.02) are also evaluated for comprehensive analysis.

### Performance Targets
- **R² Score**: > 0.8 (explains >80% of variance)
- **Tolerance Accuracy**: High percentage within ±0.01 µm
- **Convergence**: Stable training without overfitting

## Usage Instructions

### Prerequisites
```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn joblib
```

### Running the Model
1. **Ensure data files are available**:
   - `data/processed/intensity_profiles_full.csv`
   - `data/processed/parameters.csv`

2. **Execute the main script**:
   ```bash
   python gap_prediction_neural_network.py
   ```

3. **Monitor training progress**:
   - Training metrics printed every 10 epochs
   - Automatic early stopping when validation loss plateaus
   - Best model automatically saved

4. **Review results**:
   - Training curves: `plots/training_history.png`
   - Evaluation plots: `plots/evaluation_results.png`
   - Detailed metrics: `results/evaluation_results.json`

### Model Loading for Inference
```python
import torch
import joblib

# Load model
model = GapPredictionCNN()
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()

# Load scaler
scaler = joblib.load('models/scaler.pkl')

# Make predictions
intensity_profile = scaler.transform(your_intensity_data.reshape(1, -1))
prediction = model(torch.FloatTensor(intensity_profile))
```

## Results Interpretation

### Training Curves Analysis
- **Loss Curves**: Monitor for overfitting (validation loss increasing while training decreases)
- **R² Progress**: Should steadily increase toward target of 0.8
- **Tolerance Accuracy**: Should improve as model learns gap patterns
- **Learning Rate**: Automatic reduction when validation loss plateaus

### Evaluation Plots Analysis
1. **Predictions vs Actual**: Points should cluster around diagonal line
2. **Error Distribution**: Should be centered around zero with small spread
3. **Absolute Error vs Actual**: Check for systematic biases across gap ranges
4. **Tolerance Accuracy**: Compare performance at different tolerance levels

### Performance Indicators
- **R² > 0.8**: Model explains >80% of gap parameter variance
- **High Tolerance Accuracy**: Most predictions within ±0.01 µm
- **Low MAE/RMSE**: Small average prediction errors
- **Stable Training**: No overfitting, smooth convergence

## Technical Implementation Details

### Key PyTorch Functions Explained
- **nn.Conv1d**: 1D convolution for feature extraction from intensity profiles
- **nn.BatchNorm1d**: Normalizes activations for stable training
- **nn.AdaptiveAvgPool1d**: Global average pooling for dimension reduction
- **F.relu**: ReLU activation function for non-linearity
- **nn.Dropout**: Randomly zeros elements during training for regularization

### Data Flow
1. **Input**: Raw intensity profile (1000 features)
2. **Preprocessing**: StandardScaler normalization
3. **Feature Extraction**: Convolutional layers with residual blocks
4. **Dimensionality Reduction**: Global average pooling
5. **Regression**: Fully connected layers for gap prediction
6. **Output**: Single gap parameter value

## Challenges and Solutions

### Challenge 1: Overfitting Prevention
**Solution**: 
- Dropout layers (0.3, 0.2)
- Early stopping with patience
- Batch normalization
- Weight decay in optimizer

### Challenge 2: Gradient Flow in Deep Networks
**Solution**: 
- Residual blocks with skip connections
- Proper weight initialization
- Batch normalization

### Challenge 3: Learning Rate Optimization
**Solution**: 
- ReduceLROnPlateau scheduler
- Automatic learning rate reduction
- Monitoring validation loss

### Challenge 4: Evaluation Methodology
**Solution**: 
- Tolerance-based accuracy metrics
- Multiple tolerance levels
- Comprehensive error analysis

## Future Improvements

1. **Data Augmentation**: Add noise injection for robustness
2. **Ensemble Methods**: Combine multiple models for better predictions
3. **Hyperparameter Tuning**: Systematic optimization of architecture parameters
4. **Cross-Validation**: K-fold validation for more robust evaluation
5. **Transfer Learning**: Pre-training on related intensity profile tasks

## Training Results

### Initial Model Performance
The first training run completed with the following results:

- **R² Score**: 0.1104 (Target: > 0.8) ❌
- **Tolerance Accuracy (±0.01)**: 5.05% ❌
- **Mean Absolute Error**: 0.359307 µm
- **Training Duration**: 50 epochs (early stopping)

### Analysis of Results
The initial model showed several issues:
1. **Low R² Score**: Only 11% of variance explained (target was 80%)
2. **Poor Tolerance Accuracy**: Only 5% of predictions within ±0.01 µm
3. **Early Plateau**: Training stopped improving after 30 epochs
4. **High Error Rate**: 36% average prediction error

### Identified Problems
1. **Data Complexity**: 1000-feature profiles may contain noise
2. **Architecture Complexity**: Residual blocks may be over-engineering
3. **Hyperparameter Issues**: Learning rate and batch size suboptimal
4. **Preprocessing**: StandardScaler may not be ideal for this data

### Improvements Implemented
Based on the analysis, an improved version (`improved_gap_prediction.py`) was created with:

1. **Data Truncation**: Reduced to 600 points (based on successful previous models)
2. **Simplified Architecture**: Removed residual blocks, simpler CNN
3. **Better Hyperparameters**: Lower learning rate (0.0005), smaller batch size (16)
4. **Enhanced Training**: Gradient clipping, increased patience (30 epochs)
5. **Improved Regularization**: Higher dropout rates, better weight decay

## Files Generated

### Models
- `models/best_model.pth` - Original trained model
- `models/improved_best_model.pth` - Improved model (when trained)

### Visualizations
- `plots/training_history.png` - Training curves and metrics
- `plots/evaluation_results.png` - Evaluation analysis plots

### Documentation
- `TRAINING_ANALYSIS.md` - Detailed analysis of training results
- `improved_gap_prediction.py` - Improved model implementation
- `test_model.py` - Model testing utilities

## Usage Instructions

### Running the Original Model
```bash
python gap_prediction_neural_network.py
```

### Running the Improved Model
```bash
python improved_gap_prediction.py
```

### Testing Trained Models
```bash
python test_model.py
```

## Lessons Learned

1. **Start Simple**: Complex architectures may not always be better
2. **Data Analysis First**: Understanding data characteristics is crucial
3. **Iterative Improvement**: Use training results to guide improvements
4. **Comprehensive Evaluation**: Multiple metrics provide better insights
5. **Documentation**: Detailed analysis enables better future decisions

## Recommendations for Future Work

### Immediate Next Steps
1. Train the improved model and compare results
2. Implement feature engineering on intensity profiles
3. Try ensemble methods combining multiple models
4. Experiment with different preprocessing techniques

### Advanced Improvements
1. Physics-informed neural networks incorporating domain knowledge
2. Attention mechanisms to focus on important profile regions
3. Multi-task learning for both gap and L_ecran prediction
4. Uncertainty quantification for prediction confidence

## Conclusion

This project demonstrates a complete neural network development pipeline from initial implementation to analysis and improvement. While the initial results didn't meet the target performance (R² > 0.8), the comprehensive analysis identified specific issues and provided a roadmap for improvement.

The infrastructure created (training pipeline, evaluation metrics, visualization tools) provides a solid foundation for rapid iteration and improvement. The detailed documentation and analysis ensure that future work can build effectively on these results.

**Key Achievements:**
- ✅ Complete neural network implementation
- ✅ Comprehensive evaluation framework
- ✅ Detailed performance analysis
- ✅ Identified improvement strategies
- ✅ Created improved model version
- ✅ Extensive documentation

**Performance Status:**
- ❌ R² Score: 0.1104 (target > 0.8)
- ❌ Tolerance Accuracy: 5.05% (needs improvement)
- ✅ Training Infrastructure: Complete and functional
- ✅ Analysis Framework: Comprehensive and detailed
