# Credit Score Classification Model
A machine learning project to predict credit default based on user data, using a Random Forest Classifier. This repository contains the complete workflow from data preprocessing and handling class imbalance to model training, evaluation, and visualization.

# 🌟 Project Overview
The primary goal of this project is to build a reliable classification model that can predict whether a customer is likely to default on a loan. The model is trained on a credit score dataset and leverages a RandomForestClassifier for its predictive power. Special attention is given to handling the common issue of class imbalance in financial datasets by using RandomOverSampler.

# ✨ Key Features
Data Preprocessing: Cleans and prepares the dataset for modeling, including one-hot encoding for categorical features.
Class Imbalance Handling: Utilizes RandomOverSampler from imbalanced-learn to create a balanced training set, improving model performance on the minority class.
Feature Scaling: Applies StandardScaler to normalize numerical features.
Model Training: Employs a fine-tuned RandomForestClassifier for robust and accurate predictions.
Comprehensive Evaluation: Measures model performance using:
Accuracy Score
Precision, Recall, and F1-Score (via Classification Report)
ROC-AUC Score
Visualization: Generates and displays a Receiver Operating Characteristic (ROC) curve to visually assess the model's performance.

# 💾 Dataset
The model is trained on the credit_score.csv dataset. This file must be present in the root directory of the project.

Target Variable: DEFAULT - A binary indicator (0 or 1) representing whether a customer has defaulted.

Features: Various customer attributes, including financial information and a categorical feature CAT_GAMBLING.

# 📊 Results
After running the script, you will see the following performance metrics. The results indicate a high-performing model with strong predictive capabilities.

Overall Accuracy: 96.78%

ROC-AUC Score: 0.9958

Classification Report
Class

Precision

Recall

F1-Score

Support

0

0.96

0.97

0.97

375

1

0.97

0.96

0.97

375

accuracy





0.97

750

macro avg

0.97

0.97

0.97

750

weighted avg

0.97

0.97

0.97

750

(Note: Your results may vary slightly due to the random state in the train-test split.)

🤖 Model Details
The core of this project is a Random Forest Classifier, an ensemble learning method that operates by constructing a multitude of decision trees at training time. It was chosen for its high accuracy, robustness to overfitting, and ability to handle both numerical and categorical data.

The model was configured with the following key hyperparameters:

n_estimators=200

max_depth=20

min_samples_split=5

min_samples_leaf=2

class_weight='balanced'
