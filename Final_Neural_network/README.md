# Final Neural Network - Simulation Data Processing

**Author:** Oussama GUELFAA  
**Date:** 01/08/2025

## Overview

This folder contains the processed simulation data for neural network training. The original MATLAB file has been processed to truncate the first 250 points from each intensity profile and saved in a compressed NumPy format for efficient loading.

## Files

### Data Files
- `simulation_processed_truncate250_start.npz` - Processed simulation data (121.78 MB)
- `experimental_processed_interp_to_sim_grid.npz` - Processed experimental data (0.13 MB)
- `simulation_processed_with_labels.npz` - **Complete training dataset** (55.89 MB) ⭐

### Scripts
- `process_simulation_data.py` - Main simulation data processing script
- `process_experimental_data.py` - Experimental data processing and interpolation script
- `verify_processed_data.py` - Simulation data verification script
- `verify_experimental_data.py` - Experimental data verification script

### Outputs
- `verification_plots.png` - Sample plots showing processed simulation data
- `sim_vs_exp_profiles.png` - Comparison between simulation and experimental profiles
- `detailed_sim_vs_exp_comparison.png` - Detailed statistical comparison plots
- `ring_structure_comparison.png` - Focused ring structure analysis
- `README.md` - This documentation file

## Data Processing Details

### Source Data
- **Original file:** `Neural_Network_Gap_Lecran_Prediction/data/raw/Train/all_banque_new_04_07_25_NEW_full.mat`
- **Source variables:** 
  - `I_subs`: (161, 140, 1000) - 3D intensity data
  - `x`: (1, 1000) - X-axis/radius values

### Processing Steps
1. **Data Extraction:** Loaded `I_subs` and `x` variables from MATLAB file
2. **Reshaping:** Reshaped `I_subs` from (161, 140, 1000) to (22540, 1000)
3. **X-axis Replication:** Replicated x-axis data for all profiles
4. **Truncation:** Removed first 250 points from each profile
5. **Compression:** Saved to compressed .npz format

### Final Data Structure

```
simulation_processed_truncate250_start.npz:
├── X_data: (22540, 750)      # Truncated intensity profiles
└── x_positions: (22540, 750) # Corresponding x-axis values
```

### Data Statistics

**Intensity Data (X_data):**
- Number of profiles: 22,540
- Points per profile: 750 (originally 1000, truncated by 250)
- Value range: 0.0497 to 1.5678
- Mean: 0.8650
- Standard deviation: 0.2776

**X-axis Data (x_positions):**
- Range: 1.7307 to 6.9160 µm
- Mean: 4.3234 µm
- Consistent across all profiles ✓

## Usage

### Loading the Data

```python
import numpy as np

# Load processed data
data = np.load('simulation_processed_truncate250_start.npz')
X_data = data['X_data']          # Shape: (22540, 750)
x_positions = data['x_positions'] # Shape: (22540, 750)

print(f"Loaded {X_data.shape[0]} profiles with {X_data.shape[1]} points each")
```

### Data Format
- **X_data:** Intensity profiles (ratio values) after truncation
- **x_positions:** Corresponding radial positions in micrometers

## Experimental Data Processing

### Source Data
- **Original file:** `Neural_Network_Gap_Lecran_Prediction/data/raw/Test/profile_exp_PS_3um_z_positive.mat`
- **Source variables:**
  - `I_profiles`: (50, 184) - Experimental intensity profiles
  - `r_exp`: (184,) - X-axis values in meters

### Processing Steps
1. **Data Loading:** Extracted experimental profiles and x-axis data
2. **Unit Conversion:** Converted r_exp from meters to micrometers (×1e6)
3. **Grid Alignment:** Used simulation x-grid as target for interpolation
4. **Range Clipping:** Clipped simulation grid to experimental range
5. **Interpolation:** Used np.interp to interpolate experimental data onto simulation grid
6. **Data Saving:** Saved to compressed .npz format with float32 precision

### Final Experimental Data Structure

```
experimental_processed_interp_to_sim_grid.npz:
├── X_data: (50, 750)      # Interpolated experimental intensity profiles
└── x_positions: (750,)    # Simulation x-grid used for interpolation
```

### Experimental Data Statistics

**Intensity Data:**
- Number of profiles: 50
- Points per profile: 750 (interpolated from 184 original points)
- Value range: 0.1344 to 1.7505
- Mean: 0.9917
- Standard deviation: 0.2878

**X-axis Alignment:**
- Range: 1.7307 to 6.9160 µm (same as simulation)
- Perfect alignment with simulation grid ✓

### Loading Experimental Data

```python
import numpy as np

# Load experimental data
exp_data = np.load('experimental_processed_interp_to_sim_grid.npz')
exp_X_data = exp_data['X_data']          # Shape: (50, 750)
exp_x_positions = exp_data['x_positions'] # Shape: (750,)

print(f"Loaded {exp_X_data.shape[0]} experimental profiles")
```

## Combined Training Dataset

### Complete Dataset with Labels
- **File:** `simulation_processed_with_labels.npz` ⭐
- **Source:** Combination of simulation data + labels.csv
- **Size:** 55.89 MB (float32 precision)

### Dataset Structure

```
simulation_processed_with_labels.npz:
├── X_data: (22540, 750)      # Intensity profiles (float32)
├── x_positions: (22540, 750) # X-axis values (float32)
└── y_data: (22540, 2)        # Target parameters (float32)
```

### Target Parameters (y_data)

**Column 0 - Gap Parameter:**
- Range: 0.005 to 0.700 µm
- Mean: 0.353 ± 0.202 µm
- Unique values: 140

**Column 1 - L_ecran Parameter:**
- Range: 8.000 to 12.000 µm
- Mean: 10.000 ± 1.162 µm
- Unique values: 161

### Loading Complete Dataset

```python
import numpy as np

# Load complete training dataset
data = np.load('simulation_processed_with_labels.npz')
X_data = data['X_data']          # Shape: (22540, 750) - intensity profiles
x_positions = data['x_positions'] # Shape: (22540, 750) - x-axis values
y_data = data['y_data']          # Shape: (22540, 2) - [gap_um, L_um]

print(f"Training data: {X_data.shape[0]} samples")
print(f"Gap range: {y_data[:, 0].min():.3f} to {y_data[:, 0].max():.3f} µm")
print(f"L_ecran range: {y_data[:, 1].min():.3f} to {y_data[:, 1].max():.3f} µm")
```

## Processing Scripts

### process_simulation_data.py
Main script that:
- Loads MATLAB file and examines available variables
- Extracts and reshapes intensity data
- Truncates first 250 points from each profile
- Saves processed data to compressed .npz format

### verify_processed_data.py
Verification script that:
- Loads and analyzes the processed data
- Generates statistical summaries
- Creates sample plots for visual inspection
- Validates data consistency

## Key Features

1. **Efficient Storage:** Compressed .npz format reduces file size
2. **Consistent Format:** All profiles have same x-axis spacing
3. **Quality Control:** Verification script ensures data integrity
4. **Documentation:** Comprehensive logging of processing steps

## Next Steps

This processed data is ready for neural network training. The truncated profiles focus on the relevant portion of the diffraction rings while maintaining the essential features for gap and L_ecran parameter prediction.

## Technical Notes

- Original profiles had 1000 points each
- Truncation removes first 250 points (indices 0-249)
- Remaining 750 points cover the most informative part of the diffraction pattern
- X-axis values are preserved and correspond to the truncated intensity data
- All 22,540 profiles are included in the processed dataset
