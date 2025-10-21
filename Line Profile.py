import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_position
from skimage.draw import line
import os

# Load the images
stack1 = tiff.imread('/Users/elinestas/Desktop/VAβs/Data/1. Raw/Session 1/58283 Ch2 11.07 V.tif')
stack2 = tiff.imread('/Users/elinestas/Desktop/VAβs/Data/1. Raw/Session 2/58283 Ch2 9.08 V.tif')

# Function to calculate the straight line profile centered on the brightest point
def calculate_line_profile(image, brightest_point, length_micrometers=40):
    half_length = int((length_micrometers / 2) * 1.024)
    start = (brightest_point[0] - half_length, brightest_point[1])
    end = (brightest_point[0] + half_length, brightest_point[1])
    rr, cc = line(start[0], start[1], end[0], end[1])
    line_profile = image[rr, cc]
    normalized_profile = line_profile / np.max(line_profile)
    return normalized_profile, start, end

# Slices to analyze
slices_to_analyze = [10, 81, 159]

for slice_index in slices_to_analyze:
    # Find the brightest point and calculate line profiles for both stacks
    brightest_point1 = maximum_position(stack1[slice_index])
    line_profile1, start1, end1 = calculate_line_profile(stack1[slice_index], brightest_point1)
    
    brightest_point2 = maximum_position(stack2[slice_index])
    line_profile2, start2, end2 = calculate_line_profile(stack2[slice_index], brightest_point2)

    # Plot the images with the line marked
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(stack1[slice_index], cmap='gray')
    ax[0].plot([start1[1], end1[1]], [start1[0], end1[0]], color='red')
    ax[0].set_title(f'Week 5 - Slice {slice_index}', fontsize=14, fontweight='bold')
    
    ax[1].imshow(stack2[slice_index], cmap='gray')
    ax[1].plot([start2[1], end2[1]], [start2[0], end2[0]], color='red')
    ax[1].set_title(f'Week 9 - Slice {slice_index}', fontsize=14, fontweight='bold')
    
    # Save each slice figure to the desktop
    slice_output_path = f'/Users/elinestas/Desktop/slice_with_line_{slice_index}.png'
    fig.savefig(slice_output_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Plot the line profiles
    micrometer_positions = np.linspace(-20, 20, len(line_profile1))
    plt.figure(figsize=(6, 6))
    plt.plot(micrometer_positions, line_profile1, label='Week 5', color='blue')
    plt.plot(micrometer_positions, line_profile2, label='Week 9', color='orange')
    plt.xlabel('Length (μm)', fontsize=12, fontweight='bold')
    plt.ylabel('Normalized Signal Intensity', fontsize=12, fontweight='bold')
    plt.title('Normalized Line Profile', fontsize=16, fontweight='bold')
    plt.legend()
    
    # Set x and y ticks
    plt.xticks(np.linspace(-20, 20, 3), fontsize=10, fontweight='bold')  # Only 3 values on x-axis
    plt.yticks(np.linspace(0.7, 1.0, 5), fontsize=10, fontweight='bold')  # Only 5 values on y-axis and within data range
    
    # Set y-axis limits to focus on data range
    plt.ylim(0.7, 1.0)
    
    # Make axis lines thicker
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)
    
    # Save the line profile plot to the desktop
    profile_output_path = f'/Users/elinestas/Desktop/line_profile_slice_{slice_index}.png'
    plt.savefig(profile_output_path, dpi=300, bbox_inches='tight')
    
    plt.show()


