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

## 2. Financial Analysis of Annual Healthcare Costs of Cardiovascular Diseases

### Financial Overview

This financial analysis provides a interactive visual analysis of annual healthcare costs (USD) associated with various cardiovascular diseases, by sex and age group. The charts highlight cost patterns and trends of cardiovascular diseases. Both men and women share a similar cost pattern across diseases.

### Chart Summaries

### **By Sex:**

* **Women** generally have higher healthcare costs across conditions compared to men.
* **Stroke** is the most expensive condition for both men and women.
* **Other Cardiovascular Diseases** have the lowest costs for both genders.

### By Age Group:

* Healthcare costs increase with age, reaching their peak in the **"80 years or older"** group.
* **Stroke** incurs the highest cost across all age groups.

---

### Data Visualization Tools

* **Python** : Used `pandas` for data manipulation alongside `plotly` for interactive charts.
* **Chart Types** :
* Sunburst Chart
* Grouped Bar Chart
* Bubble Chart

This analysis is useful and provides the ability to predict cardiovascular diseases which can help save lives and significantly reduce the stress of its financial burden.

Resource: In resource folder, file "supplemental material" pg.15

## **3. Logistic Regression Model**

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

## **4. Linear Regression Model**

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

![residualplot](Challenges\project4_team2\Visualizations\linear_residual_plot.png)
![regressionplot](Challenges\project4_team2\Visualizations\logistic_regression_curve.png)

---

## **5. Neural Network Model**

### Overview

This project focuses on developing and optimizing a neural network model using TensorFlow and KerasTuner to achieve high accuracy on the heart failue dataset. Key aspects include hyperparameter tuning, manual optimization, and evaluation of the model's performance on test data.
Neural Network Model

    Framework: TensorFlow
    Features: All columns in the dataset were used as features. The dataset contains no categorical data.

### Auto Optimization

Methodology

    Tool: KerasTuner was used to optimize hyperparameters for the neural network model.
    Performance Metrics:
        Best Validation Accuracy: 0.893
        Total Elapsed Time: 3 minutes 24 seconds![autoparams](Challenges\project4_team2\Visualizations\auto_params.png)

alt text

Insights:

    The highest validation accuracy achieved was 0.893.
    The second and third highest validation accuracies were 0.880 and 0.867, respectively.
    Mean validation accuracy across trials was 0.698 (denoted by the red dashed line).

This plot highlights significant variation in validation accuracy across trials, with some configurations performing much better than others.

### Best Hyperparameters

{
  "activation": "tanh",
  "first_units": 63,
  "num_layers": 3,
  "units_0": 3,
  "units_1": 5,
  "units_2": 5,
  "units_3": 3,
  "units_4": 5,
  "tuner/epochs": 20,
  "tuner/initial_epoch": 0,
  "tuner/bracket": 0,
  "tuner/round": 0
}

    Total Parameters: 1,067

alt text
Results on Test Dataset

    Loss: 0.400
    Accuracy: 0.893

### Manual Optimization

Optimization Attempts and Results

    Using Auto-Optimized Parameters:
        Architecture: Layers with 63, 3, 5, 5, and 1 units (all using tanh activation)
        Parameters: 1,067
        Results: Loss = 0.556, Accuracy = 0.853

    Testing Activation Functions:
        Architecture: Layers with 50, 100, and 1 units (activations: relu, relu, sigmoid)
        Parameters: 5,851
        Results: Loss = 0.355, Accuracy = 0.880

    Testing LeakyReLU Activation:
        Architecture: Layers with 50, 100, and 1 units (activations: LeakyReLU, relu, sigmoid)
        Parameters: 5,851
        Results: Loss = 0.388, Accuracy = 0.853

    Increased Epochs (20 → 50):
        Same architecture as Attempt #2
        Results: Loss = 0.605, Accuracy = 0.800

    Preventing Overfitting:
        Parameters: 5,851
        Results: Loss = 0.514, Accuracy = 0.840

    Cross-Validation:
        Results: TBD

    Addressing Overfitting Warnings:
        Architecture: Adjusted activations with LeakyReLU and relu
        Results: Loss = 0.517, Accuracy = 0.840

### Model Evaluation

Loss Curve
![losscurve](Challenges\project4_team2\Visualizations\model_loss.png)
alt text

    Training Loss: The plot shows the model loss during training and validation over the course of 20 epochs. The blue line represents the training loss, which starts high and then decreases rapidly over the first few epochs before leveling off. The orange line represents the validation loss, which also starts high and decreases over time, but not as dramatically as the training loss. The overall trend shows that the model is learning effectively, with the training loss and validation loss both declining as the number of training epochs increases.

Accuracy Curve
![modelaccuracy](Challenges\project4_team2\Visualizations\model_accuracy.png)
alt text

Training Accuracy: The image shows a plot of model accuracy over multiple epochs. There are two lines on the plot:

The blue line represents the training accuracy of the model. This shows how well the model is performing on the training data as the training progresses over the epochs. The orange line represents the validation accuracy of the model. This shows how well the model is generalizing to unseen data, i.e., the validation data, as the training progresses.

The plot illustrates how the training accuracy and validation accuracy of the model change over the course of 15 epochs. Overall, the training accuracy appears to be increasing, while the validation accuracy shows more fluctuation but a general upward trend as well.

### Key Takeaways

    Learning Dynamics:
        Initial overfitting was mitigated with adjustments.
        Training loss and accuracy trends indicate effective learning from the data.
        Validation performance highlights room for improvement in generalization.

    Final Test Metrics:
        Loss: 0.517
        Accuracy: 0.840
        These metrics reflect the true performance on unseen data.

    Optimization Insights:
        Hyperparameter tuning played a critical role in achieving high validation accuracy.
        Manual optimizations provided insights into the impact of activation functions and overfitting prevention techniques.

### Conclusion

This deep learning model highlights the iterative process of training, optimizing, and evaluating a neural network. Through a combination of automated and manual tuning, the model achieved a test accuracy of 0.893, demonstrating its potential to perform effectively on unseen data while maintaining strong generalization. Given its application in the early detection of individuals at risk for cardiovascular diseases, the model shows promise as an initial screening tool. However, further improvements are essential to enhance its accuracy and reliability.
Recommended next steps include:

    Expanding the training dataset to improve model robustness.
    Analyzing the data for outliers and addressing inconsistencies.
    Integrating related datasets to introduce new features that may enhance predictive performance.

## **6. Gradient Boosting Classifier**

### Gradient Boosting Classifier - An Introduction

Gradient Boosting is a powerful boosting algorithm that combines several weak learners into strong learners, in which each new model is trained to minimize the loss function such as mean squared error or cross-entropy of the previous model using gradient descent. In each iteration, the algorithm computes the gradient of the loss function with respect to the predictions of the current ensemble and then trains a new weak model to minimize this gradient. The predictions of the new model are then added to the ensemble, and the process is repeated until a stopping criterion is met. Source: https://www.geeksforgeeks.org/ml-gradient-boosting/

### Setup

In order to visualize the decision tree, you will need to install/download graphviz on your machine. See here for instructions on how to download graphviz. You may also need to install the following libraries in your VS code environment:

    xgboost !Pip install xgboost
    Graphviz !Pip install Graphviz

### Results

For this analysis, two separate libraries were used to train gradient boosting models: GradientBoostingClassifier from sklearn and XGBClassifier. The analysis below will breakdwon the results from both models. It is import to note that prior to training the different models, the dataset was scaled using standard scaler.

Please note the following nomencleture used in the analysis below:

    death event - patient is deceased
    no-death event - patient is not deceased

### GradientBoostingClassifier

    Acccuracy score - The model as a whole had a accuracy score of 89% when predicting death events
    Precision - Based on the clasification report, we can tell that 90% of the predicted no-death events were correct and 88% of the predicted death events were correct. This indicates that the precision of this model is quite high
    Recall - No-death events had a recall score of 96% while no death events had a recall score of 88%. This indicates that for death events, the ratio of correctly predicted death events to the total actual death events was a 88%. By comparison, the recall for death events was 70% indicating the model is not as good at prediting when there will be no deaths

### XGBClassifier

    Acccuracy score - The model as a whole had a accuracy score of 91% when predicting death events
    Precision - Based on the clasification report, we can tell that 94% of the predicted no-death events were correct and 81% of the predicted death events were correct. This indicates that the precision of this model is quite high
    Recall - No-death events had a recall score of 96% while no death events had a recall score of 93%. This indicates that for death events, the ratio of correctly predicted death events to the total actual death events was a 88%. By comparison, the recall for death events was 85%.

### Conclusion

In conclusion, even though both libraries are used to train gradient boosting classifier models, XGBOOST was a far better fit for our data set. It was able to more accurately predict both death and non-death events
