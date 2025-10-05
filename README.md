# Predictive Modeling of the Calgary Real Estate Market

This repository contains the complete data science project for developing a predictive model for residential property assessed values in Calgary, Alberta. The project demonstrates an end-to-end workflow, from sourcing and cleaning raw public data to training, evaluating, and interpreting an advanced machine learning model.

## Table of Contents

1.  [Project Objective](https://www.google.com/search?q=%23project-objective)
2.  [Methodology](https://www.google.com/search?q=%23methodology)
3.  [Results & Key Findings](https://www.google.com/search?q=%23results--key-findings)
4.  [How to Run](https://www.google.com/search?q=%23how-to-run)
5.  [Tools and Libraries](https://www.google.com/search?q=%23tools-and-libraries)

-----

## Project Objective

The goal of this project was to bring data-driven transparency to the often opaque real estate market. By leveraging publicly available data, this analysis aimed to build a robust model that can accurately predict property values and identify the key factors that influence them, serving as an open-source tool for public benefit.

-----

## Methodology

The project was developed in a Google Colab environment and followed a structured, iterative workflow.

#### 1\. Data Sourcing & Preparation

The primary dataset was the "Property Assessment" data from the City of Calgary's Open Data portal, containing over 2.2 million records. Initial preprocessing in Pandas involved correcting data types for key numerical columns and filtering the dataset to a working set of only "Residential" properties.

#### 2\. Feature Engineering

To create a powerful location feature, geospatial coordinates were engineered from the `MULTIPOLYGON` column. The centroid of each property's boundary polygon was calculated using the Shapely library to generate precise `latitude` and `longitude` features.

#### 3\. Modeling & Evaluation

An iterative modeling strategy was employed:

  * A baseline **Linear Regression** model was established, which proved insufficient for this complex task (R-squared of 0.13).
  * An **XGBoost Regressor** was implemented, which significantly improved performance.
  * Model diagnostics, including **feature importance analysis** and an evaluation of error metrics (MAE vs. RMSE), revealed that a small percentage of high-value outliers were disproportionately skewing the results.
  * The final model was trained on a filtered dataset that excluded these outliers (properties assessed over $3 million), allowing it to specialize on the core residential market. The model's performance was validated on a 20% held-out test set.

-----

## Results & Key Findings

The final, refined XGBoost model achieved an **R-squared of 0.73** on the test set, indicating it successfully explains 73% of the variance in Calgary's core residential property market.

  * **Key Insight 1: Geospatial Features are Crucial:** Using precise `latitude` and `longitude` coordinates was a far more effective location feature than using categorical neighborhood names, boosting the R-squared score from 0.39 to 0.50.
  * **Key Insight 2: Outlier Handling is Critical:** The most significant performance gain came from strategically filtering out a small fraction (\<1%) of extreme high-value properties. This increased the final R-squared from 0.50 to 0.73 and resulted in a more robust and reliable model.
  * **Key Insight 3: A Well-Fitting Model:** The final model's performance on the test set (R²: 0.73) was nearly identical to its performance on the train set (R²: 0.73), confirming that the model learned a general pattern and was not simply memorizing the data.

-----

## How to Run

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    ```
2.  **Set up the Data:**
      * Download the "Property Assessment Data" CSV from the City of Calgary's Open Data portal.
      * Place the CSV file in a Google Drive folder at the path `My Drive/MIAI_Project/data/`.
3.  **Open in Google Colab:**
      * Upload the `.ipynb` notebook file from this repository to your Google Colab account.
      * Ensure the runtime is set to use a `T4 GPU` for efficient training.
4.  **Run the Notebook:** Execute the cells in the notebook from top to bottom.

-----

## Tools and Libraries

  * **Language:** Python
  * **Core Libraries:** Pandas, Scikit-learn, XGBoost, Shapely
  * **Environment:** Google Colab (T4 GPU)
  * **Version Control:** Git / GitHub
