import numpy as np
import h5py
from skimage import filters, measure, morphology
import napari

# Paths
h5_file_path = '/Users/elinestas/Desktop/VAβs/Data/3. Preprocessed/Plaques/Test.1.1.h5'
output_file_path = '/Users/elinestas/Desktop/VAβs/Desktop'

# Load the image stack from the HDF5 file
with h5py.File(h5_file_path, 'r') as f:
    image_stack = f['image_stack'][:]

# Conversion pixel to um
pixel_size_um = 500 / 512  # Pixel size in micrometers
slice_thickness_um = 5  # Thickness between slices in micrometers
area_conversion_factor = pixel_size_um ** 2
volume_conversion_factor = pixel_size_um ** 2 * slice_thickness_um

# Minimum plaque size
min_size_um2 = 40
min_size_px2 = min_size_um2 / area_conversion_factor

# Function to perform post-processing calculations
def post_process(viewer, label_layer):
    # Calculate volume and average depth after manual correction
    labeled_stack = label_layer.data
    regions = measure.regionprops(labeled_stack)
    plaque_data = []

    for region in regions:
        area = region.area * area_conversion_factor
        volume = region.area * volume_conversion_factor
        centroid = np.array(region.centroid)
        average_depth = np.mean([coord[0] for coord in region.coords]) * slice_thickness_um

        plaque_data.append({
            'Volume (um^3)': volume,
            'Average Depth (um)': average_depth
        })

    # Convert plaque data to structured array
    dtype = [('Volume (um^3)', 'f8'), ('Average Depth (um)', 'f8')]
    plaque_array = np.array([tuple(d.values()) for d in plaque_data], dtype=dtype)

    # Save the plaque data and processed stack to an npy file
    output_data = {
        'filtered_stack': labeled_stack.astype(np.uint8) * 255,
        'plaque_parameters': plaque_array
    }

    np.save(output_file_path, output_data)
    print("Processing complete and data saved.")
    viewer.close()

# Ask the user if they want to proceed with automatic analysis
proceed = input("Do you want to proceed with automatic analysis? (yes/no): ").strip().lower()
if proceed == 'yes':
    # User input for start and end slices where you see plaques when looking through the stack
    start_slice = int(input("Enter the start slice number where you see plaques: "))
    end_slice = int(input("Enter the end slice number where you see plaques: "))
    if start_slice < 0 or end_slice >= image_stack.shape[0] or start_slice > end_slice:
        raise ValueError("Invalid slice range. Please enter valid start and end slice numbers.")

    # 3D Segmentation on the stack
    binarised_stack = np.zeros_like(image_stack, dtype=bool)

    for i in range(start_slice, end_slice + 1):
        image = image_stack[i]
        
        # 4 classes for multi-level thresholding
        num_classes = 4
        
        # Multi-level thresholding to define thresholds specific to plaque characteristics
        thresholds = filters.threshold_multiotsu(image, classes=num_classes)
        regions = np.digitize(image, bins=thresholds)
        
        # Consider both class 3 and class 4 as plaques
        plaques = (regions == 3) | (regions == 4)
        binary = plaques  # Convert to binary image
        
        # Morphological operations to remove small objects and fill small holes
        binary_cleaned = morphology.remove_small_objects(binary, min_size=min_size_px2)
        binary_cleaned = morphology.remove_small_holes(binary_cleaned, area_threshold=min_size_px2)
        
        # Store the cleaned binary image in the processed stack
        binarised_stack[i] = binary_cleaned

    # Perform 3D connected component labeling on the entire stack
    labeled_stack, num_features = measure.label(binarised_stack, connectivity=3, return_num=True)

    # Filter out small regions and regions that appear in less than 4 or more than 40 slices
    regions = measure.regionprops(labeled_stack)
    filtered_stack = np.zeros_like(labeled_stack)

    for region in regions:
        area = region.area * area_conversion_factor
        slice_extent = np.max(region.coords[:, 0]) - np.min(region.coords[:, 0]) + 1
        
        if min_size_px2 <= region.area and 2 <= slice_extent <= 40:
            for coord in region.coords:
                filtered_stack[coord[0], coord[1], coord[2]] = region.label
else:
    filtered_stack = np.zeros_like(image_stack, dtype=np.uint8)

# Open the stack in napari for manual annotation or review
viewer = napari.Viewer(ndisplay=2)  # Set ndisplay to 2 for 2D view
viewer.add_image(image_stack, name='Original Stack')
label_layer = viewer.add_labels(filtered_stack, name='Filtered Labeled Stack')

# Add a hook to the close event to run the post-process function
viewer.window._qt_window.closeEvent = lambda event: (post_process(viewer, label_layer), event.accept())

# Run napari and allow the user to correct the labels manually
napari.run()
