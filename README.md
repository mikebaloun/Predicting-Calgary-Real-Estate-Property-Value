# Predictive ML Modeling of the Calgary Real Estate Market

This repository contains the complete, end-to-end data science project for developing a highly accurate predictive model for residential property assessed values in Calgary, Alberta. The project demonstrates a full, iterative workflow, from sourcing and cleaning raw public data to advanced feature engineering, hyperparameter tuning, and model interpretation.

The final tuned XGBoost model achieves an **R-squared score of 0.84**, successfully explaining 84% of the variance in the core residential market.

## Table of Contents

1.  [Project Objective](https://www.google.com/search?q=%23project-objective)
2.  [Data Source](https://www.google.com/search?q=%23data-source)
3.  [Methodology](https://www.google.com/search?q=%23methodology)
4.  [Results & Key Findings](https://www.google.com/search?q=%23results--key-findings)
5.  [How to Run](https://www.google.com/search?q=%23how-to-run)
6.  [Tools and Libraries](https://www.google.com/search?q=%23tools-and-libraries)

-----

## Project Objective

The goal of this project was to bring data-driven transparency to the often opaque real estate market. By leveraging publicly available data, this analysis aimed to build a robust model that can accurately predict property values and identify the key factors that influence them, serving as an open-source tool for public benefit.

-----

## Data Source

The primary dataset for this project is the **Property Assessment Data** from the City of Calgary's Open Data portal. This comprehensive dataset contains over 2.2 million historical assessment records, including property characteristics, location information, and the assessed value used for taxation.

The data can be accessed directly from the portal: [Calgary Property Assessment Data](https://data.calgary.ca/Government/Total-Property-Assessed-Value/dmd8-bmxh)

-----

## Methodology

The project was developed in a Google Colab environment and followed a structured, iterative workflow.

#### 1\. Data Preparation

The raw dataset was loaded into Pandas, where it was cleaned to correct data types and filtered to include only residential properties.

#### 2\. Feature Engineering

Geospatial coordinates were engineered from the `MULTIPOLYGON` column. The centroid of each property's boundary was calculated using the Shapely library to generate precise `latitude` and `longitude` features.

#### 3\. Modeling and Optimization

An iterative modeling strategy was employed. To ensure the model could generalize to new data and to prevent overfitting, the dataset was partitioned into an **80% training set and a 20% held-out test set**.

  * An **XGBoost Regressor** was selected as the primary algorithm.
  * The dataset was strategically filtered to focus the model on the core residential market.
  * **Hyperparameter tuning** was performed using `RandomizedSearchCV` to find the optimal settings for the model, which was the final step in maximizing its predictive accuracy.

-----

## Results & Key Findings

The final, tuned XGBoost model demonstrated excellent performance and a strong, generalizable fit.

#### Final Tuned Model Performance

| Metric | Test Set (Unseen Data) | Train Set (Seen Data) |
| :--- | :--- | :--- |
| **R-squared** | 0.836 | 0.845 |
| **MAE** | $72,556.72 | $71,431.93 |
| **RMSE** | $122,021.68 | $118,516.21 |

The model's performance on the training set was nearly identical to the test set, confirming that the model is **well-fitted and not "memorizing" the data**.

  * **Key Insight 1: Hyperparameter Tuning is Crucial:** The final tuning step provided the most significant performance boost, increasing the R-squared score from 0.73 to 0.84.
  * **Key Insight 2: Geospatial Coordinates are Superior:** Using precise `latitude` and `longitude` was a more effective location feature than simple categorical labels.
  * **Key Insight 3: Outlier Handling is Key:** Strategically removing a small fraction of extreme outliers was critical to building a reliable and accurate model for the general market.

-----

## How to Run

1.  **Clone the Repository:** `git clone https://github.com/your-username/your-repository-name.git`
2.  **Set up the Data:** Download the "Property Assessment Data" CSV from the link in the Data Source section and place it in a Google Drive folder at the path `(https://drive.google.com/drive/folders/1pFD7AK32eBGZV5wry9PpK3Dd4mWuSmTg?usp=drive_link)`.
3.  **Open in Google Colab:** Upload the `.ipynb` notebook file to Google Colab and ensure the runtime is set to `T4 GPU`.
4.  **Run the Notebook:** Execute the cells in the notebook from top to bottom.

-----

## Tools and Libraries

  * **Language:** Python
  * **Core Libraries:** Pandas, Scikit-learn, XGBoost, Shapely
  * **Environment:** Google Colab (T4 GPU)
  * **Version Control:** Git / GitHub
