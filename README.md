# Calgary Real Estate Valuation Engine 🏘️

**A residential Automated Valuation Model (AVM) achieving 94.3% R² on unseen data.**

---

## 📊 Executive Summary

This project implements an end-to-end machine learning pipeline to predict **residential property values in Calgary, Alberta** using open municipal data.

Moving beyond simple geospatial analysis, the model uses a **Hybrid Spatial–Categorical strategy** to capture the nuance of urban pricing. By engineering features around property utility and neighborhood clusters, it achieves high predictive performance with a clean generalization profile.

### Key Performance Indicators (Held-out Test Set)

| Metric | Score | Context |
| :--- | :--- | :--- |
| **R² Score** | **0.943** | Explains 94.3% of variance in assessed values. |
| **MAE** | **$41,951** | Average absolute error in dollar terms (~8% of value). |
| **RMSE** | **$80,392** | Penalizes larger errors more heavily. |
| **Generalization** | **1.0%** | Minimal gap between Train vs. Test scores (Low overfitting). |

---

## 🧠 Engineering Strategy: "Context over Coordinates"

Early baselines using only raw Latitude/Longitude features plateaued around **0.84 R²**. The jump to **0.94 R²** came from treating **context** as a first-class signal:

1.  **Granular Zoning (`SUB_PROPERTY_USE`)**
    Allows the model to distinguish between property types (e.g., "Low-rise Condo" vs. "Detached Home") instead of treating all residential lots as homogeneous.
2.  **Neighborhood Codes (`COMM_CODE`) with Native Categoricals**
    Uses `enable_categorical=True` in XGBoost to handle high-cardinality neighborhood identifiers directly, avoiding memory-heavy one-hot encoding.
3.  **Spatial Centroids from Geometry**
    Parses WKT `MULTIPOLYGON` data via **Shapely** to extract precise geometric centroids (longitude/latitude) for each parcel.

Combined, these features let the model learn: *what the property is*, *where it is*, and *how it’s used*—not just raw coordinates.

---

## 🗂️ Data Pipeline

* **Source:** City of Calgary Open Data
* **Dataset:** "Total Property Assessed Value"
* **Scope:** Filtered to `ASSESSMENT_CLASS_DESCRIPTION == "Residential"`

**Preprocessing highlights:**
* **Cleaning:** Parsed currency-like strings and filtered explicitly for valid residential zoning.
* **Outlier Handling:** Restricted to values < $3M to stabilize training on the general market.
* **Geometry:** Extracted centroids from `MULTIPOLYGON` shapes using `shapely.wkt.loads`.

---

## 📉 Visual Diagnostics

### 1. Feature Importance
*The model correctly identifies Property Type and Neighborhood as the primary drivers of value, matching human appraisal logic.*
![Feature Importance](images/Tuned%20Model%20Feature%20Importance.png)

### 2. Actual vs. Predicted Values (Density Heatmap)
*A tight cloud along the 45° line indicates strong predictive accuracy. The heatmap reveals high density around the $400k–$700k market segment.*
![Actual vs Predicted](images/Actual%20vs%20Predicted%20Values.png)

### 3. Residual Analysis
*Errors are centered near zero and approximately symmetric. The "fan" shape reflects natural heteroscedasticity—larger dollar errors for more expensive homes—rather than systematic bias.*
![Residuals](images/Residuals.png)

### 4. Correlation Matrix
*Core numeric inputs exhibit low to moderate correlation, which supports model stability.*
![Correlation Matrix](images/Correlation%20Matrix.png)

---

## ⚙️ Tech Stack

* **Language:** Python 3.10+
* **Core Libraries:** `pandas`, `numpy`
* **Modeling:** `xgboost.XGBRegressor` (GPU-accelerated)
* **Geospatial:** `shapely` (WKT parsing)
* **Visualization:** `seaborn`, `matplotlib`

---

## 🚀 How to Run

1.  **Clone the repo**
    ```bash
    git clone [https://github.com/mikebaloun/Predicting-Calgary-Real-Estate-Property-Value.git](https://github.com/mikebaloun/Predicting-Calgary-Real-Estate-Property-Value.git)
    cd Predicting-Calgary-Real-Estate-Property-Value
    ```

2.  **Install dependencies**
    ```bash
    pip install xgboost shapely pandas numpy seaborn matplotlib scikit-learn
    ```

3.  **Download the Data**
    * Download the "Total Property Assessed Value" CSV from the [City of Calgary Open Data portal](https://data.calgary.ca/).
    * Place it in a `data/` directory.

4.  **Run the Analysis**
    * Upload `Predicting_Calgary_Real_Estate_Property_Value.ipynb` to Google Colab.
    * Select a **T4 GPU** runtime for faster execution.
    * Run all cells to reproduce the pipeline.

---
*Author: Michael Baloun | Date: November 17, 2025*
