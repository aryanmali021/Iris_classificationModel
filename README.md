## Project2 decodelabs

# Iris Flower Classification Using KNN

A Machine Learning classification project that predicts the species of an Iris flower using sepal and petal measurements. The project uses the K-Nearest Neighbors (KNN) algorithm from Scikit-Learn.

## Features

* Loads the Iris dataset
* Performs train-test splitting
* Applies feature scaling
* Trains a K-Nearest Neighbors (KNN) classifier
* Evaluates model performance
* Generates Accuracy, F1 Score, and Confusion Matrix
* Saves the trained model for future predictions

## Dataset

The Iris dataset contains:

* 150 flower samples
* 3 flower species:

  * Setosa
  * Versicolor
  * Virginica
* 4 features:

  * Sepal Length
  * Sepal Width
  * Petal Length
  * Petal Width

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Joblib

## Project Structure

```text
Iris_classificationModel/
│
├── data/
│   └── iris.csv
│
├── models/
│   └── trained_model.pkl
│
├── main.py
├── requirements.txt
├── report.txt
└── README.md
```

## Model Performance

Accuracy: 100%

F1 Score: 100%

Confusion Matrix:

```text
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

## Learning Outcomes

* Supervised Learning
* Data Preprocessing
* Feature Scaling
* Train-Test Split
* K-Nearest Neighbors (KNN)
* Model Evaluation
* Model Persistence

## Author

Aryan Mali
B.Tech Artificial Intelligence & Data Science


