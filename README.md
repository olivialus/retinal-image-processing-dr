# Retinal Image Enhancement for Diabetic Retinopathy Screening

Python image-processing research exploring classical and quantum-inspired techniques for enhancing retinal images used in diabetic retinopathy screening.

## Overview

This project was developed through Cal Poly's Summer Undergraduate Research Program (SURP). The research focused on improving retinal vessel visibility and evaluating image-processing methods that could support downstream machine-learning analysis for diabetic retinopathy detection.

This work contributed to the paper **"Quantum Inspired Retinal Enhancement for Diabetic Retinopathy Detection,"** accepted to **IEEE IEMCON 2026**.

## Image Processing Pipeline

The included Python pipeline applies:

- Green-channel extraction
- CLAHE contrast enhancement
- Gaussian smoothing
- Frequency-domain processing using an FFT-based, quantum-inspired weighting method
- Blending of processed and original retinal imagery

## Technologies

- Python
- NumPy
- SciPy
- OpenCV
- scikit-image
- Matplotlib
- Image Processing
- Machine Learning

## Research Scope

During the project, I:

- Developed and evaluated 10+ classical and quantum-inspired retinal image-processing pipelines
- Worked with the DRIVE and APTOS 2019 retinal image datasets
- Extracted 100+ image features for machine-learning analysis
- Evaluated neural network, gradient boosting, random forest, and logistic regression models

The strongest model achieved:

- **AUC:** 0.931
- **Accuracy:** 76.6%
- **F1 Score:** 0.759
- **MCC:** 0.639

## Repository Contents

`retinal_enhancement_pipeline.py`  
Implements the retinal image-enhancement pipeline and visualization workflow.

`requirements.txt`  
Lists the Python dependencies required to run the project.

## Publication

This research contributed to work accepted to **IEEE IEMCON 2026**.

## Data

The retinal image datasets used in this research are not included in this repository. The project used publicly available retinal datasets including DRIVE and APTOS 2019.

## Author

**Olivia Lusignan**  
Computer Science, Cal Poly San Luis Obispo  
Bioinformatics Minor
