

# CellTracker AI: Segmentation, Tracking & Feature Extraction and Analysis

**CellTracker AI** is a smart microscopy analysis system that performs **cell segmentation**, **temporal cell tracking**, **mitosis detection**, and **quantitative feature extraction and analysis** from raw microscopy videos.  
It enables **automated single-cell behavior analysis** and **dataset generation** for downstream research, modelling, and machine learning.

This project was developed for **quantitative live-cell imaging workflows** and supports generating **visual tracking videos**, **labeled masks**, and **feature tables** that can be directly analyzed or integrated into ML models.

---

## Key Features

| Feature | Description |
|--------|-------------|
| **Segmentation** | Uses **Mask R-CNN (ResNet50-FPN)** for instance-level cell segmentation. |
| **Cell Tracking** | Tracks cells across frames using detection-to-track matching with IoU assignment. |
| **Mitosis Detection** | Automatically identifies cell division events. |
| **Feature Extraction & Analysis** | Computes motion, morphology, size dynamics, and acceleration metrics per tracked cell. |
| **Dataset Export** | Saves extracted cell-level measurements into structured CSV files. |
| **Visualization** | Generates labeled tracking videos with stable color-coded cell identities & Visualize the extracted features|

---

## Applications

- Live-cell imaging analytics  
- Quantitative single-cell motility and growth studies  
- Automated lineage tracking  
- Cell behavior profiling and clustering  
- Dataset generation for ML-based cell phenotype classification  
- Computational biology and biomedical image analysis research  

---

## Models Tested During Development

| Model | Purpose |
|------|---------|
| **Cellpose (`model_type="nuclei"`)** | Segmentation |
| **U-Net++ (`encoder=resnet34`)** | Segmentation |
| **Mask R-CNN (selected)** | Final segmentation choice |

---

## Dashboard Overview

- View segmented or raw video
- Inspect single cell trajectories
- Visualize the extracted features
- Detect anomalies using IsolationForest
- Compare cell behaviors by role/state
- Explore lifetime and movement patterns
- Export cleaned datasets for ML or statistics

---

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a01ea48b-2146-4479-af40-c78d259d221b" />

---

<img width="1920" height="1035" alt="image" src="https://github.com/user-attachments/assets/500868ab-2268-485f-a4ef-51be5bac7849" />

---

<img width="1919" height="1080" alt="image" src="https://github.com/user-attachments/assets/14f22517-7ccd-47d1-b71c-7617ff26050f" />

---
