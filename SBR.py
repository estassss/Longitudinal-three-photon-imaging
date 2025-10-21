import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Define the file paths for the vessels and plaques TIFF files
file_paths = {
    'Vessels': {
        'Week 5': {
            'mouse1': '/Users/elinestas/Desktop/VAβs/Data/1. Raw/Session 1/58283 Ch2 11.07 V.tif'
        },
        'Week 9': {
            'mouse1': '/Users/elinestas/Desktop/VAβs/Data/1. Raw/Session 2/58283 Ch2 9.08 V.tif'
        }
    },
    'Plaques': {
        'Week 5': {
            'mouse1': '/Users/elinestas/Desktop/VAβs/Data/1. Raw/Session 1/58283 Ch1 11.07 V.tif'
        },
        'Week 9': {
            'mouse1': '/Users/elinestas/Desktop/VAβs/Data/1. Raw/Session 2/58283 Ch1 9.08 V.tif'
        }
    }
}

# Function to calculate SBR
def calculate_sbr(signal_img):
    """Calculate the Signal-to-Background Ratio for a given depth."""
    peak_value = np.max(signal_img)
    background_first_15 = np.mean(signal_img.flatten()[:15])
    background_last_15 = np.mean(signal_img.flatten()[-15:])
    background = (background_first_15 + background_last_15) / 2
    sbr = peak_value / (background + 1)
    return sbr

# Function to calculate the signal strength as the average of the brightest 0.5% pixels
def calculate_signal_strength(signal_img):
    """Calculate the signal strength for a given depth."""
    sorted_pixels = np.sort(signal_img.flatten())
    threshold_index = int(0.995 * len(sorted_pixels))
    signal_strength = np.mean(sorted_pixels[threshold_index:])
    return signal_strength

# Initialize a list to store SBR and signal strength data
sbr_data = []
signal_strength_data = []

# Process each session and calculate SBR and signal strength for vessels and plaques
for session in file_paths['Vessels']:
    vessel_paths = file_paths['Vessels'][session]
    plaque_paths = file_paths['Plaques'][session]

    for mouse in vessel_paths:
        vessel_path = vessel_paths[mouse]
        plaque_path = plaque_paths[mouse]

        vessel_img = tiff.imread(vessel_path)
        plaque_img = tiff.imread(plaque_path)

        # Limit the range to the minimum depth of the two images to avoid index errors
        max_depth = min(vessel_img.shape[0], plaque_img.shape[0])
        
        for depth in range(max_depth):
            vessel_sbr = calculate_sbr(vessel_img[depth])
            plaque_sbr = calculate_sbr(plaque_img[depth])
            vessel_signal_strength = calculate_signal_strength(vessel_img[depth])
            plaque_signal_strength = calculate_signal_strength(plaque_img[depth])
            
            sbr_data.append({
                'Session': f'Plaques {session}',
                'Depth': depth * 5,  # Assuming each slice represents 5 micrometers in depth
                'SBR': plaque_sbr
            })
            sbr_data.append({
                'Session': f'Vessels {session}',
                'Depth': depth * 5,
                'SBR': vessel_sbr
            })
            
            signal_strength_data.append({
                'Session': f'Plaques {session}',
                'Depth': depth * 5,
                'Signal_Strength': np.log(plaque_signal_strength)
            })
            signal_strength_data.append({
                'Session': f'Vessels {session}',
                'Depth': depth * 5,
                'Signal_Strength': np.log(vessel_signal_strength)
            })

# Convert the data to DataFrames
sbr_df = pd.DataFrame(sbr_data)
signal_strength_df = pd.DataFrame(signal_strength_data)

# Plot SBR per depth
plt.figure(figsize=(10, 6))

sns.lineplot(data=sbr_df, x='Depth', y='SBR', hue='Session',
             palette=['cyan', 'magenta', 'cyan', 'magenta'], style='Session', markers=False,
             dashes=[(2, 2), (2, 2), '', ''])

plt.xlabel('Depth (μm)', fontsize=14, fontweight='bold')
plt.ylabel('SBR', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()

# Make axis lines
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)

# Save the SBR plot
plt.savefig('/Users/elinestas/Desktop/sbr_per_depth.png', dpi=300)
plt.show()

# Plot Signal Strength per depth
plt.figure(figsize=(10, 6))

sns.lineplot(data=signal_strength_df, x='Depth', y='Signal_Strength', hue='Session',
             palette=['cyan', 'magenta', 'cyan', 'magenta'], style='Session', markers=False,
             dashes=[(2, 2), (2, 2), '', ''])

plt.xlabel('Depth (μm)', fontsize=14, fontweight='bold')
plt.ylabel('ln Signal (normalized unit)', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.tight_layout()

# Make axis lines
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)

# Save the signal strength plot
plt.savefig('/Users/elinestas/Desktop/signal_strength_per_depth.png', dpi=300)
plt.show()
