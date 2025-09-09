import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Dataset
try:
    dataset = pd.read_csv("credit_score.csv")
    print("Dataset loaded successfully.\n")
except FileNotFoundError:
    print("Error: The file 'credit_score.csv' was not found.")
    print("Please make sure it is in the same directory as this script.")
    exit()

# drop unwanted column
df_cleaned = dataset.drop(columns=[dataset.columns[0]])

# one hot encoding
df_encoded = pd.get_dummies(df_cleaned, columns=["CAT_GAMBLING"], dtype=int)

# Features and Target
x = df_encoded.drop(columns=["DEFAULT"])
y = df_encoded["DEFAULT"]

# balancing the output of dataset
balance = RandomOverSampler()
x_balance, y_balance = balance.fit_resample(x, y)

# Train test split 70 % training and 30 % testing
x_train, x_test, y_train, y_test = train_test_split(
    x_balance, y_balance, test_size=0.3, random_state=42, stratify=y_balance
)

# feature scaling
scaler = StandardScaler()
# We apply scaling only to numerical columns
num_column = x_train.select_dtypes(include=["int64", "float64"]).columns
x_train[num_column] = scaler.fit_transform(x_train[num_column])
x_test[num_column] = scaler.transform(x_test[num_column])
print("Feature scaling complete.\n")


# model training with random forest classifier
print("Training the RandomForestClassifier model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

# model fit
model.fit(x_train, y_train)

print("Model training complete.\n")

# accuracy and proba
y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:, 1]

# Calculate and print the overall accuracy.
accuracy = model.score(x_test, y_test)
print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

# Generate and print the Classification Report (Precision, Recall, F1-Score).
print("--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Calculate and print the ROC-AUC score.
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")


# --- 8. Visualize the ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

# Create and save a plot for the ROC Curve.
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot(
    [0, 1], [0, 1], color="red", linestyle="--", label="Random guess (area = 0.50)"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
