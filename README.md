**Title:** A Predictive Model for the Calgary Real Estate Market: An End-to-End Data Science Project
**Author:** Michael Baloun
**Date:** October 5, 2025

---

### **Executive Summary**

This report details the development of a highly accurate predictive model for residential property assessed values in Calgary, Alberta. The project demonstrates a full, iterative workflow, from sourcing and cleaning a large-scale public dataset of over 2.2 million records to advanced feature engineering, hyperparameter tuning, and model interpretation. A key innovation was the extraction of geospatial coordinates from raw polygon data to create a superior location feature. The final, tuned XGBoost model achieves an **R-squared score of 0.84**, successfully explaining 84% of the variance in the core residential market. This project serves as a comprehensive case study in applying a robust data science methodology to a complex, real-world problem.

---

### **1.0 Introduction**

The goal of this project was to bring data-driven transparency to the often opaque real estate market. By leveraging publicly available data, this analysis aimed to build a robust model that can accurately predict property values and identify the key factors that influence them, serving as an open-source tool for public benefit.

### **2.0 Methodology**

The project was developed in a Google Colab environment and followed a structured, iterative workflow.

**2.1 Data Preparation**
The primary dataset, sourced from the City of Calgary's Open Data portal, was loaded into Pandas for cleaning. This involved correcting data types, handling formatting inconsistencies, and filtering the data to a working set of residential properties.

**2.2 Feature Engineering**
Geospatial coordinates were engineered from the `MULTIPOLYGON` column. The centroid of each property's boundary was calculated using the Shapely library to generate precise `latitude` and `longitude` features, which proved to be more effective than categorical location labels.

**2.3 Modeling and Optimization**
To ensure the model could generalize to new data, the dataset was partitioned into an 80% training set and a 20% held-out test set.
* An **XGBoost Regressor** was selected as the primary algorithm.
* The dataset was strategically filtered to focus the model on the core residential market (properties assessed under $3 million) after diagnostics revealed that high-value outliers were skewing performance.
* **Hyperparameter tuning** was performed using `RandomizedSearchCV` to find the optimal settings for the model, which was the final step in maximizing its predictive accuracy.

### **3.0 Results and Discussion**

The final, tuned XGBoost model demonstrated excellent performance and a strong, generalizable fit.

| Metric | Test Set (Unseen Data) | Train Set (Seen Data) |
| :--- | :--- | :--- |
| **R-squared** | 0.836 | 0.845 |
| **MAE** | $72,556.72 | $71,431.93 |
| **RMSE** | $122,021.68 | $118,516.21 |

The close alignment between the test and train set scores confirms that the model is **well-fitted and not "memorizing" the data**. Key insights from the analysis include the critical importance of hyperparameter tuning, the superiority of geospatial coordinate features, and the significant impact of strategic outlier removal.

### **4.0 Conclusion**

This project successfully culminated in a high-performing predictive model for a large and complex dataset. The process demonstrates a complete, end-to-end data science workflow, from data sourcing and cleaning to advanced modeling and interpretation. The final model and the documented process serve as a strong testament to the practical application of machine learning techniques.

---

### **Appendix A: Replication Steps**

1.  **Clone the Repository:** `git clone https://github.com/your-username/your-repository-name.git`
2.  **Set up the Data:** Download the "Property Assessment Data" CSV from [this link](https://data.calgary.ca/Government/Total-Property-Assessed-Value/dmd8-bmxh) and place it in a Google Drive folder at the path `My Drive/MIAI_Project/data/`.
3.  **Open in Google Colab:** Upload the `.ipynb` notebook file to Google Colab and ensure the runtime is set to `T4 GPU`.
4.  **Run the Notebook:** Execute the cells in the notebook from top to bottom.

### **Appendix B: Tools and Libraries**

* **Language:** Python
* **Core Libraries:** Pandas, Scikit-learn, XGBoost, Shapely
* **Environment:** Google Colab (T4 GPU)
* **Version Control:** Git / GitHub
