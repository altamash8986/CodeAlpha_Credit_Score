# 🏦 Credit Scoring Model (Random Forest)

This project builds a **credit scoring model** using **Random Forest Classifier** to predict whether a customer will default on credit.  
It handles **imbalanced datasets** with oversampling, applies **feature scaling**, and evaluates performance using **ROC-AUC** and other classification metrics.  

The project includes full **data preprocessing, model training, evaluation, and visualization of the ROC Curve**.  

---

## 🚀 Features

- ✅ **Data Cleaning & Preprocessing** (dropping unwanted columns, one-hot encoding)  
- ✅ **Handling Imbalanced Data** with `RandomOverSampler` from imbalanced-learn  
- ✅ **Feature Scaling** with `StandardScaler`  
- ✅ **Model Training** using `RandomForestClassifier` with optimized hyperparameters  
- ✅ **Evaluation Metrics**:  
  - Accuracy  
  - Precision, Recall, F1-Score (classification report)  
  - ROC-AUC Score  
  - ROC Curve visualization  

---

## 📂 Project Structure

- ├── credit_score.csv # Dataset (not included in repo for privacy, add your own) 
- ├── credit_scoring.py # Main Python script
- ├── README.md # Project documentation

## 🚗 Run Output
python credit_scoring.py


## 📊 Example Output

Overall Accuracy: 87.45%

--- Classification Report ---
              precision    recall  f1-score   support
           0       0.89      0.86      0.87       300
           1       0.85      0.88      0.86       280

ROC-AUC Score: 0.9321

## 🧠 Model Insights

The dataset was imbalanced, meaning defaulters (class 1) were underrepresented.
→ Fixed using RandomOverSampler for fairer training.

Random Forest with class_weight="balanced" ensures that both classes are considered equally important.

ROC-AUC is prioritized over just accuracy, since in credit risk management, catching defaulters is critical.

## 👨‍💻 Author

👤 Name: Mohd Altamash
📧 Email: mohdaltamash37986@gmail.com

🌐 LinkedIn: https://www.linkedin.com/in/mohd-altamash-0997592a6?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app

💻 GitHub: https://github.com/altamash8986
