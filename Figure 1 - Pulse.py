import numpy as np
import matplotlib.pyplot as plt
import os

# Load data from the file, skipping the header rows
file_path = '/Users/elinestas/Desktop/Pulse.txt'
data = np.loadtxt(file_path, skiprows=23)

# Extract columns
delay = data[:, 0]  # Delay in ps
intensity = data[:, 1]  # Intensity in arbitrary units
fit = data[:, 2]  # Fitted data

# Plot the data
plt.figure(figsize=(4, 4))
plt.plot(delay, intensity, 'bo-', markersize=3, label="Measured Pulse")
plt.plot(delay, fit, 'r-', linewidth=2, label="Fitted Curve")

plt.xlabel("Delay (ps)")
plt.ylabel("Intensity (arb. units)")
plt.grid(False)

folder_path = '/Users/elinestas/Desktop/3P paper'
image_path = os.path.join(folder_path, "Pulse.svg")
plt.savefig(image_path, format="pdf")
plt.show()
