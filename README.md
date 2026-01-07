# Satellite Imagery Based Property Valuation

This project estimates residential property prices by combining **structured housing attributes** with **satellite imagery features**. The approach leverages computer vision for spatial context and machine learning regression for price prediction.

---

## Project Objective

The objective of this project is to improve property valuation accuracy by integrating:
- Traditional tabular housing features (area, rooms, location, etc.)
- Environmental and neighborhood context extracted from satellite images

Instead of training deep CNNs end-to-end, pretrained CNN models are used **only for feature extraction**, ensuring computational efficiency and stability.

---

## Repository Structure

| File | Description |
|-----|------------|
| `data_fetcher.py` | Downloads satellite images using Mapbox Static Images API based on latitude and longitude |
| `preprocessing.ipynb` | Data cleaning, EDA, feature engineering, and CNN-based image embedding extraction |
| `model_training.ipynb` | Regression models and Grad-CAM visualizations |
| `X_train_image_embeddings.npy` | Precomputed image embeddings (train) |
| `X_test_image_embeddings.npy` | Precomputed image embeddings (test) |
| `X_train_tabular.csv` | Processed tabular training features |
| `X_test_tabular.csv` | Processed tabular test features |
| `y_train.csv` | Target variable (training prices) |
| `y_val.csv` | Validation targets |
| `enrollno_final.csv` | Final prediction file (submission format) |
| `README.md` | Project documentation |

---

## Methodology Overview

### 1. Satellite Image Collection
- Satellite images are fetched using **Mapbox Static Images API**
- Images are centered around property coordinates
- Fixed zoom and resolution ensure consistency

### 2. Image Feature Extraction (Preprocessing Stage)
- A **pretrained ResNet18 CNN** is used
- CNN is applied **only once** to extract image embeddings
- Extracted embeddings represent spatial and neighborhood features
- Embeddings are saved as `.npy` files for reuse

> This design avoids repeated CNN execution during model training, making the pipeline efficient and stable.

### 3. Tabular Feature Engineering
- Missing values handled
- Outliers treated
- Duplicate Rows checked
- Feature scaling applied
- Relevant predictors selected using correlation analysis

### 4. Model Approach
- Tabular regression model
- Image-only regression model (using embeddings)
- Fused regression model (tabular + image embeddings)

The fused approach provides the best balance between interpretability and predictive performance.

---

## Grad-CAM Interpretability

Grad-CAM is applied to the pretrained CNN to visualize regions of satellite images that contribute most to learned embeddings.  
This provides visual interpretability by highlighting:
- Surrounding green areas
- Road networks
- Building density
- Plot layout

Grad-CAM is used on **selected sample images only**, ensuring minimal computational overhead.

---

## Results Summary

- The fused model achieves strong predictive performance with an **R² score of approximately 0.86**
- Image-only models show limited performance individually, confirming that satellite imagery works best when combined with tabular data
- Predictions are stable and consistent across validation and test sets

---

## Submission Notes

- CNN models are **not trained end-to-end**
- Image embeddings are **precomputed**
- Kernel crashes are avoided
- All outputs are reproducible
- Final predictions are stored in `enrollno_final.csv` with strict format:
  id, predicted_price

---

## Tools & Libraries

- Python
- NumPy, Pandas
- PyTorch (pretrained CNN only)
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Mapbox Static Images API

---

## Author

**Payili Meenakshi**  
Satellite Imagery Based Property Valuation Project  
January 2026  

GitHub: https://github.com/payilimeenakshi2
