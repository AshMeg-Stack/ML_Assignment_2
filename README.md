ML Assignment 2 – Heart Disease Classification
a) Problem Statement
The objective of this assignment is to implement and evaluate multiple machine learning classification models to predict the presence of heart disease. The project demonstrates model comparison, performance evaluation using multiple metrics, and deployment through a Streamlit web application.
b) Dataset Description
Dataset Name: Heart Disease Dataset
Source: Publicly available dataset (UCI / Kaggle)
Number of Instances: 1025
Number of Input Features: 13
Target Variable: Binary classification (0 = No Heart Disease, 1 = Heart Disease)
The dataset satisfies assignment requirements of having more than 12 features and more than 500 instances.
c) Models Used
•	Logistic Regression
•	Decision Tree
•	K-Nearest Neighbors (KNN)
•	Naive Bayes
•	Random Forest (Ensemble Model)
•	XGBoost (Ensemble Model)
Evaluation Metrics Used:
•	Accuracy
•	AUC (Area Under Curve)
•	Precision
•	Recall
•	F1 Score
•	Matthews Correlation Coefficient (MCC)
•	Confusion Matrix
•	Classification Report
d) Comparison Table
ML Model	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.8098	0.9298	0.7619	0.9143	0.8312	0.6309
Decision Tree	0.9854	0.9857	1.0000	0.9714	0.9855	0.9712
KNN	0.8439	0.9453	0.8230	0.8857	0.8532	0.6891
Naive Bayes	0.8293	0.9043	0.8070	0.8762	0.8402	0.6602
Random Forest	1.0000	1.0000	1.0000	1.0000	1.0000	1.0000
XGBoost	1.0000	1.0000	1.0000	1.0000	1.0000	1.0000
e) Observations on Model Performance
Logistic Regression (Accuracy: 0.81, F1: 0.83)
Performs reasonably well but significantly lower than tree-based models. This suggests the dataset contains non-linear relationships that a linear model cannot fully capture.

Decision Tree (Accuracy: 0.99, F1: 0.99)
Achieves very high performance, indicating strong separability in the dataset. However, near-perfect metrics suggest potential overfitting when using a single tree.

KNN (Accuracy: 0.84, F1: 0.85)
Moderate improvement over Logistic Regression shows presence of local feature similarity patterns, but it does not generalize as strongly as ensemble models.

Naive Bayes (Accuracy: 0.83, F1: 0.84)
Comparable to KNN but slightly lower, indicating that independence assumptions limit its ability to model interacting features.

Random Forest (Accuracy: 1.00, F1: 1.00)
Perfect performance suggests the dataset has strong feature interactions that ensemble tree methods capture effectively. More stable compared to a single Decision Tree.

XGBoost (Accuracy: 1.00, F1: 1.00)
Matches Random Forest performance, confirming that boosting methods are highly effective for this dataset’s structure.

Live Streamlit Application
https://mlassignment2-7rpc73nvpygtwxpieoujq2.streamlit.app/
GitHub Repository
https://github.com/AshMeg-Stack/ML_Assignment_2
