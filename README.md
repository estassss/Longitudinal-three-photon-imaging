# Longitudinal three-photon imaging analysis  
Code for the manuscript:  
**“Longitudinal three-photon imaging for tracking amyloid plaques and vascular degeneration in a mouse model of Alzheimer’s disease.”**

## Overview
This repository contains the analysis code used to process three-photon microscopy data of vasculature and amyloid plaques in APP NL-G-F mice imaged twice over a four-week interval. The scripts support preprocessing, vessel and plaque segmentation, quantitative metric extraction, and reproduction of manuscript figures.

## Repository contents

### Figure scripts (Python)
- Figure 1 – Pulse.py  
- Figure 1 – Resolution.py  
- Figure 3 – Visualise GFP neurons.py  
- Figure 3 – Visualise with THG.py  
- Figure 4 – Line Profile.py  
- Figure 4 – SBR.py  
- Figure 5 – Volumes.py  
- Figure 6 – Violin Plot.py  
- Figure Sup. A (a) & (b) - Power Calibration & Motion.py

### Preprocessing and segmentation
- Normalise_Vessels.py  
- Plaque Seg – Preprocess.py  
- PlaqueSeg – Semi-Automatic Annotation.py  
- Removecrosstalk.m  

### Vessel architecture (MATLAB)
- Vessel_Architecture.m – Computes vessel diameter, length, tortuosity, and inter-vessel distance.

**Vessel segmentation note:**  
Vessel segmentation note:  
Vessel masks were generated using the DeepVess pipeline:  
Haft-Javaherian M., Fang L., Muse V., Schaffer C.B., Nishimura N., Sabuncu M.R. “Deep convolutional neural networks for segmenting 3D in vivo multiphoton images of vasculature in Alzheimer disease mouse models.” *PLOS ONE* 14(3): e0213539, 2019. doi:10.1371/journal.pone.0213539

## Data summary
The data is accessible on DataDryad (https://doi.org/10.5061/dryad.wh70rxx2j) and contains TIFF image stacks (16-bit) acquired with three-photon excitation at 1340 nm.  
- XY pixel size: 0.98 µm  
- Z-step: 5 µm (vessels/plaques), 10 µm (GFP neurons)  
- Labels: Texas Red (vessels), methoxy-XO4 (plaques), GFP (neurons), THG when present  
- Depth: up to ~900 µm  
Additional stacks of neonatal lung stacks for power calibration.

## Usage
1. Download TIFF files from the associated Dryad dataset.  
2. Clone this repository:  
   ```bash
   git clone https://github.com/estassss/Longitudinal-three-photon-imaging
