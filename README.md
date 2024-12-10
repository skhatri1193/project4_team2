# Advanced Machine Learning: Heart Failure Dataset Analysis

This project analyzes the likelihood of death due to heart failure, survival time prediction, and associated healthcare costs. By leveraging advanced machine learning models and visualization tools, we aim to provide a comprehensive understanding of heart failure outcomes, survival analysis, and the financial burden associated with cardiovascular diseases.

---

## **1. Introduction to the Dataset**

The dataset ( *heart_failure_clinical_records_dataset.csv* ) contains clinical and demographic data about heart failure patients, including:

* Age
* Blood Pressure
* Creatinine Levels
* Ejection Fraction
* Death Event (DEATH_EVENT)

This dataset supports multiple machine learning models to analyze survival time, predict death events, and assess healthcare costs related to cardiovascular conditions.

---

## **2. Logistic Regression Model**

### **Objective:**

Predict the likelihood of death (binary classification).

### **Key Steps:**

* Checked and cleaned the dataset (no missing values found).
* Standardized features using `StandardScaler`.
* Trained and tested using an 80-20 split.
* Evaluated using a confusion matrix, classification report, and ROC-AUC.

### **Performance:**

* **Accuracy:** 80%
* **AUC (ROC):** 0.83
* **Key Findings:** The model shows strong predictive power but could miss some deaths due to limitations in recall for death events.

---

## **3. Linear Regression Model**

### **Objective:**

Predict survival time (continuous variable).

### **Key Steps:**

* Focused on relationships between age, serum creatinine, and survival time.
* Evaluated using metrics like Mean Squared Error (MSE) and R-squared (R²).

### **Performance:**

* **MSE:** 0.93
* **R²:** -0.02 (indicating poor fit)

### **Insights:**

The residuals showed patterns, suggesting the linear model struggled to capture the complexity of survival time predictions.

---

## **4. Neural Network Model**

### **Objective:**

Optimize predictions of heart failure events using deep learning.

### **Methodology:**

* Framework: TensorFlow, with hyperparameter optimization using `KerasTuner`.
* Key Metrics:
  * **Validation Accuracy (Best):** 0.893
  * **Test Accuracy:** 0.893
  * **Test Loss:** 0.400

### **Optimization Results:**

* Auto-tuned and manually adjusted architectures for better accuracy.
* Best manual configuration achieved **Accuracy: 0.880** and  **Loss: 0.355** .

---

## **5. Gradient Boosting Classifier**

### **Objective:**

Predict death events with high accuracy using boosting techniques.

### **Models:**

* **GradientBoostingClassifier** :
* **Accuracy:** 89%
* **Recall (Deaths):** 88%
* **XGBClassifier** :
* **Accuracy:** 91%
* **Recall (Deaths):** 93%
* **Precision (Deaths):** 81%

### **Conclusion:**

XGBClassifier demonstrated better overall performance for this dataset, making it the preferred model for predicting death events.

---

## **6. Financial Overview**

This financial analysis provides an interactive visual analysis of annual healthcare costs (USD) associated with various cardiovascular diseases, segmented by sex and age group.

### **Key Insights:**

#### **By Sex:**

* Women generally have higher healthcare costs compared to men.
* Stroke is the most expensive condition for both genders.
* Other cardiovascular diseases incur the lowest costs.

#### **By Age Group:**

* Costs increase with age, peaking in the "80 years or older" group.
* Stroke is the costliest condition across all age groups.

---

### **Data Visualization Tools:**

* **Python** : Used `pandas` for data manipulation and `plotly` for interactive charts.
* **Chart Types** :
* Sunburst Chart
* Grouped Bar Chart
* Bubble Chart

### **Conclusion:**

These insights highlight cost patterns and trends for cardiovascular diseases, providing valuable information for predicting disease risks and alleviating the financial burden.
