import napari
from skimage import io, img_as_float
import numpy as np

# Load the image
image_path = '/Users/elinestas/Desktop/VAβs/Data/GFP neurons.tif'
stack = io.imread(image_path)
stack = img_as_float(stack)
if stack.shape[0] == 2:
    stack = np.transpose(stack, (1, 0, 2, 3))  # → (Z, C, H, W)

# Set image spacing
z_spacing = 10  # microns
xy_spacing = 500 / 512  # microns per pixel

# Create Napari viewer
viewer = napari.Viewer(ndisplay=3)

# Add image with proper scaling
viewer.add_image(
    stack,
    channel_axis=1,
    colormap=['magenta', 'green'],
    name=['Channel 1', 'Channel 2'],
    blending='additive',
    scale=(z_spacing, xy_spacing, xy_spacing),
)


viewer.theme = 'light'
viewer.camera.angles = (0, 45, 180)
viewer.camera.zoom = 1.5
viewer.scale_bar.visible = True
viewer.scale_bar.unit = 'µm'


napari.run()
