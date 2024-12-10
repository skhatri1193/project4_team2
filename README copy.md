# team2_project4
Project 4: Advanced Machine Learning
LINEAR REGRESSION:

The aim of our linear regression is to predict the likelihood of death due to heart failure, and analyze the survival time as a continuous variable. The dataset (heart_failure_clinical_records_dataset.csv) contains clinical and demographic data about heart failure patients. This dataset include patient age, blood pressure, creatinine levels, ejection fraction, and whether they experienced a death event (DEATH_EVENT).

We began by using SQL as it stimulates real-world scenario where healthcare data is stored in databases. We then checked for missing value just to make sure our dataset is complete and clean to work with - found nothing. We used StandardScaler because Age and Creatine have different ranges to scaling makes sure that all feature contribute the same to the model. We then prepared for our first regression analysis to Predict Death. We began by splitting the data into 80% going for training and 20% for testing. We then trained our logistic regression, and predicted:

accuracy: 80%

classification report:

class 0 (NO DEATH): Precision 76%, Recall 97%, F1-score 85%
Class 1 (DEATH): Precision 93%, Recall 56%, F1-score 70%.
confusion matrix:

True Positives (Deaths correctly predicted): 14
False Positives (Wrongly predicted deaths): 1
True Negatives (No death correctly predicted): 34
False Negatives (Missed deaths): 11
We visualized a confusion matrix and the results are attached below:

Screenshot 2024-12-04 at 6 47 03 PM
We improved our performance evaluation by using cross-validation scores.

We then moved on to a linear regression for Survival Time. We trained the regression to capture a relationship between features like age and serum creatinine and survival time. We decided to calculate MSE which is average squared error between pedicted and actual survival times (0.93) and R2 to be around -0.02 to tell us that the model persoms worse than predicting the mean.

We visualized residuals to see if the model’s errors are random (ideal) or patterned and they were not randomly distributed telling us that the model has problems in capturing relationship between features and the survival time: Screenshot 2024-12-04 at 6 48 27 PM

We then optimized our model's performance by tuning hyperparameters. We generated a Receiver Operating Characteristic curve, calculated the Area under the curve (AUC) to evaluate the performance of our logistic regression model. Understanding the ROC Curve:

X-axis (False Positive Rate - FPR): Represents the proportion of survivors incorrectly classified as having a death event. Lower FPR indicates fewer false positives, improving model reliability.
Y-axis (True Positive Rate - TPR): Indicates the proportion of actual death events correctly identified by the model. Higher TPR reflects better accuracy in predicting deaths.
Diagonal Line: Represents random guessing (AUC = 0.5). A model performing above this line indicates better-than-random classification. Model Performance (AUC Analysis):
AUC (Area Under the Curve) = 0.83: The model has strong predictive power, correctly distinguishing between death and survival outcomes 83% of the time.
AUC Interpretation:
AUC = 1: Perfect model.
AUC = 0.5: Random guessing.
AUC between 0.5 and 1: Indicates the model’s discriminative ability. Higher AUC values signify better performance. Conclusion: The ROC curve and the AUC value of 0.83 demonstrate that the model is highly effective in predicting heart failure outcomes. This analysis supports its use for identifying high-risk patients, aiding in better clinical decisions-making
Screenshot 2024-12-04 at 6 50 12 PM
We thought it would be ideal to get feature correlation analysis with Death Events and Survival Time in order to understand the data better and helping identify the most relevant ones. However, we found that there were only weaker correlations as the tables below show: Screenshot 2024-12-04 at 6 51 27 PM Screenshot 2024-12-04 at 6 51 37 PM

Lastly, we plotted a Logistic Regression Curve representing Age vs the Probability of Death. After scaling, age values below the mean will become negative which means these represent younger patients and values above the mean, those that are positive represent older patients. Initially the model was train on multiple features (X_Train) which added outside noise to our data. So we focused exclusively on how age related to DEATH_EVENTS. By simplifying our data in this way, we removed outside noise and were able to generate a smoother more interpretable curve rather than a straight line where the probability is either 0 or 1.

Screenshot 2024-12-04 at 6 54 47 PM
We can see that the curve is increasing with age, highlighting the probability of death and an increase in age positively correlated. In other words, as you grow older, the probability of death due to heart failure problems increase.
