import os
import pandas as pd
# pyrefly: ignore [missing-import]
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Step 1: Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Optional: Save dataset to CSV for reference
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['target'] = y
df.to_csv("data/iris.csv", index=False)
print("Saved dataset to data/iris.csv")

# Step 2: Apply Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Step 4: Create & Train KNN Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Optional: Save the trained model
joblib.dump(model, "models/trained_model.pkl")
print("Saved model to models/trained_model.pkl")

# Step 5: Predict
y_pred = model.predict(X_test)

# Step 6: Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

f1 = f1_score(y_test, y_pred, average="weighted")
print("F1 Score:", f1)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
