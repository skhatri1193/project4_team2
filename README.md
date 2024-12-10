# team2_project4

# Gradient Boosting Classifier - An Introduction 
Gradient Boosting is a powerful boosting algorithm that combines several weak learners into strong learners, in which each new model is trained to minimize the loss function such as mean squared error or cross-entropy of the previous model using gradient descent. In each iteration, the algorithm computes the gradient of the loss function with respect to the predictions of the current ensemble and then trains a new weak model to minimize this gradient. The predictions of the new model are then added to the ensemble, and the process is repeated until a stopping criterion is met. Source: https://www.geeksforgeeks.org/ml-gradient-boosting/

# Setup
In order to visualize the decision tree, you will need to install/download graphviz on your machine. See [here](https://www.graphviz.org/download/) for instructions on how to download graphviz. You may also need to install the following libraries in your VS code environment:
* xgboost
`!Pip install xgboost`
* Graphviz
`!Pip install Graphviz`


# Results
For this analysis, two separate libraries were used to train gradient boosting models: GradientBoostingClassifier from sklearn and XGBClassifier. The analysis below will breakdwon the results from both models. It is import to note that prior to training the different models, the dataset was scaled using standard scaler.

Please note the following nomencleture used in the analysis below:
* death event - patient is deceased
* no-death event - patient is not deceased


## GradientBoostingClassifier
* Acccuracy score - The model as a whole had a accuracy score of 89% when predicting death events
* Precision - Based on the clasification report, we can tell that 90% of the predicted no-death events were correct and 88% of the predicted death events were correct. This indicates that the precision of this model is quite high
* Recall -  No-death events had a recall score of 96% while no death events had a recall score of 88%. This indicates that for death events, the ratio of correctly predicted death events to the total actual death events was a 88%. By comparison, the recall for death events was 70% indicating the model is not as good at prediting when there will be no deaths

## XGBClassifier
* Acccuracy score - The model as a whole had a accuracy score of 91% when predicting death events
* Precision - Based on the clasification report, we can tell that 94% of the predicted no-death events were correct and 81% of the predicted death events were correct. This indicates that the precision of this model is quite high
* Recall -  No-death events had a recall score of 96% while no death events had a recall score of 93%. This indicates that for death events, the ratio of correctly predicted death events to the total actual death events was a 88%. By comparison, the recall for death events was 85%.


# Conclusion

In conclusion, even though both libraries are used to train gradient boosting classifier models, XGBOOST was a far better fit for our data set. It was able to more accurately predict both death and non-death events





