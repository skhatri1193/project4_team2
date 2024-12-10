
# Team 2: Predicting Heart Failure

Project 4: Advanced Machine Learning

LINEAR REGRESSION:

The aim of our linear regression is to predict the likelihood of death due to heart failure, and analyze the survival time as a continuous variable.
The dataset (heart_failure_clinical_records_dataset.csv) contains clinical and demographic data about heart failure patients. This dataset include patient age, blood pressure, creatinine levels, ejection fraction, and whether they experienced a death event (DEATH_EVENT).

We began by using SQL as it stimulates real-world scenario where healthcare data is stored in databases. We then checked for missing value just to make sure our dataset is complete and clean to work with - found nothing. We used StandardScaler because Age and Creatine have different ranges to scaling makes sure that all feature contribute the same to the model.
We then prepared for our first regression analysis to Predict Death. We began by splitting the data into 80% going for training and 20% for testing. We then trained our logistic regression, and predicted:

- accuracy: 80%
- classification report:

  - class 0 (NO DEATH): Precision 76%, Recall 97%, F1-score 85%
  - Class 1 (DEATH): Precision 93%, Recall 56%, F1-score 70%.
- confusion matrix:

  * True Positives (Deaths correctly predicted): 14
  * False Positives (Wrongly predicted deaths): 1
  * True Negatives (No death correctly predicted): 34
  * False Negatives (Missed deaths): 11

  We visualized a confusion matrix and the results are attached below:
  `<img width="420" alt="Screenshot 2024-12-04 at 6 47 03 PM" src="https://github.com/user-attachments/assets/19194377-2ca4-4b23-b731-24e549e9a647">`

We improved our performance evaluation by using cross-validation scores.

We then moved on to a linear regression for Survival Time. We trained the regression to capture a relationship between features like age and serum creatinine and survival time. We decided to calculate MSE which is average squared error between pedicted and actual survival times (0.93) and R2 to be around -0.02 to tell us that the model persoms worse than predicting the mean.

We visualized residuals to see if the model’s errors are random (ideal) or patterned and they were not randomly distributed telling us that the model has  problems in capturing relationship between features and the survival time:
`<img width="580" alt="Screenshot 2024-12-04 at 6 48 27 PM" src="https://github.com/user-attachments/assets/80b1acdc-6072-4d3e-be4e-78147703e7b1">`

We then optimized our model's performance by tuning hyperparameters. We generated a Receiver Operating Characteristic curve, calculated the Area under the curve (AUC) to evaluate the performance of our logistic regression model. Understanding the ROC Curve:

- X-axis (False Positive Rate - FPR):
  Represents the proportion of survivors incorrectly classified as having a death event. Lower FPR indicates fewer false positives, improving model reliability.
- Y-axis (True Positive Rate - TPR):
  Indicates the proportion of actual death events correctly identified by the model. Higher TPR reflects better accuracy in predicting deaths.
- Diagonal Line:
  Represents random guessing (AUC = 0.5). A model performing above this line indicates better-than-random classification.
  Model Performance (AUC Analysis):
- AUC (Area Under the Curve) = 0.83:
  The model has strong predictive power, correctly distinguishing between death and survival outcomes 83% of the time.
  - AUC Interpretation:
    - AUC = 1: Perfect model.
    - AUC = 0.5: Random guessing.
    - AUC between 0.5 and 1: Indicates the model’s discriminative ability. Higher AUC values signify better performance.
      Conclusion:
      The ROC curve and the AUC value of 0.83 demonstrate that the model is highly effective in predicting heart failure outcomes. This analysis supports its use for identifying high-risk patients, aiding in better clinical decisions-making
      `<img width="567" alt="Screenshot 2024-12-04 at 6 50 12 PM" src="https://github.com/user-attachments/assets/a3726ae8-dfac-4e4c-bceb-823cc32184e1">`

We thought it would be ideal to get feature correlation analysis with Death Events and Survival Time in order to understand the data better and helping identify the most relevant ones. However, we found that there were only weaker correlations as the tables below show:
`<img width="305" alt="Screenshot 2024-12-04 at 6 51 27 PM" src="https://github.com/user-attachments/assets/caed6ff1-87d3-481a-8e53-0ac68e23a928">` `<img width="302" alt="Screenshot 2024-12-04 at 6 51 37 PM" src="https://github.com/user-attachments/assets/9dce4ed6-02cc-4baa-911d-f7c937ad04cf">`

Lastly, we plotted a Logistic Regression Curve representing Age vs the Probability of Death. After scaling, age values below the mean will become negative which means these represent younger patients and values above the mean, those that are positive represent older patients. Initially the model was train on multiple features (X_Train) which added outside noise to our data. So we focused exclusively on how age related to DEATH_EVENTS. By simplifying our data in this way, we removed outside noise and were able to generate a smoother more interpretable curve rather than a straight line where the probability is either 0 or 1.

<img width="696" alt="Screenshot 2024-12-04 at 6 54 47 PM" src="https://github.com/user-attachments/assets/d37bbdf2-b450-4846-9177-8d001fba0d0b">

We can see that the curve is increasing with age, highlighting the probability of death and an increase in age positively correlated. In other words, as you grow older, the probability of death due to heart failure problems increase.
=========================================================================================================================================================================================================================================

# Team 2 Project 4

### **Part 1: Linear Regression and Logistic Regression**

#### Objectives:

* **Linear Regression** : Predict survival time as a continuous variable.
* **Logistic Regression** : Predict the likelihood of death (DEATH_EVENT).

#### Data Preprocessing:

* **Dataset** : Clean and complete; no missing values.
* **Scaling** : Used `StandardScaler` to normalize features with varying ranges.

#### Logistic Regression Results:

* **Accuracy** : 80%.
* **Performance Metrics** :
* Class 0 (No Death): Precision 76%, Recall 97%, F1-Score 85%.
* Class 1 (Death): Precision 93%, Recall 56%, F1-Score 70%.
* **Confusion Matrix** :
* True Positives: 14
* False Positives: 1
* True Negatives: 34
* False Negatives: 11
* **ROC Curve** :
* AUC: 0.83, indicating strong predictive power.

#### Linear Regression Results:

* **Metrics** :
* MSE: 0.93
* R2R^2**R**2: -0.02, indicating the model performed worse than predicting the mean.
* **Residual Analysis** : Non-random distribution of errors indicates model limitations.

#### Insights:

* **Feature Correlation** : Weak correlations between features and outcomes.
* **Logistic Regression Curve** : Demonstrated a clear positive relationship between age and the probability of death.

---

### **Part 2: Gradient Boosting Classifier**

#### Algorithm Overview:

* Gradient Boosting iteratively improves model performance by minimizing loss functions using gradient descent.

#### Implementation:

* **Libraries** : Used `GradientBoostingClassifier` (sklearn) and `XGBClassifier` (XGBoost).
* **Preprocessing** : Scaled data using `StandardScaler`.

#### Results:

1. **GradientBoostingClassifier** :

* Accuracy: 89%.
* Precision:
  * No-Death Events: 90%
  * Death Events: 88%
* Recall:
  * No-Death Events: 96%
  * Death Events: 70%

1. **XGBClassifier** :

* Accuracy: 91%.
* Precision:
  * No-Death Events: 94%
  * Death Events: 81%
* Recall:
  * No-Death Events: 96%
  * Death Events: 85%

#### Conclusion:

* **XGBClassifier** outperformed `GradientBoostingClassifier`, particularly in recall for predicting death events.

---

### **Part 3: Neural Networks**

#### Objective:

Develop and optimize a neural network to predict outcomes in the heart failure dataset.

#### Implementation:

* **Framework** : TensorFlow.
* **Optimization Tool** : KerasTuner.
* **Features** : All columns were included; no categorical data.

#### Auto Optimization:

* **Best Validation Accuracy** : 0.893.
* **Mean Validation Accuracy** : 0.698.
* **Best Hyperparameters** :
* Activation: `tanh`
* Layers: 3
* Units: Varying (63, 3, 5, etc.)
* Epochs: 20
* **Total Parameters** : 1,067.

#### Manual Optimization:

1. **Using Auto-Optimized Parameters** :

* Accuracy: 0.853.

1. **Testing Activation Functions** (`ReLU`, `LeakyReLU`):
   * Improved architecture resulted in Accuracy: 0.880.
2. **Increased Epochs** :

* Results pending (based on provided data).

#### Insights:

* Hyperparameter tuning significantly impacted performance.
* Activation function selection and layer architecture were critical for achieving better accuracy.

---

### **Overall Conclusion**

This project demonstrates the practical application of machine learning techniques in healthcare:

1. **Logistic Regression** provided interpretable results and insights into mortality predictors.
2. **Gradient Boosting** enhanced classification accuracy, with XGBoost outperforming.
3. **Neural Networks** showed potential for higher accuracy with thorough hyperparameter optimization.

The findings have direct implications for healthcare decision-making, particularly in identifying high-risk patients and improving clinical outcomes. Future work could explore additional features, external datasets, or ensemble approaches to further refine predictions.
