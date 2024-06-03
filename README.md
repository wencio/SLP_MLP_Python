

# Titanic Survival Analysis and Perceptron Implementation

## Project Overview
This project involves analyzing the Titanic dataset to predict passenger survival using a perceptron model. The notebook covers data loading, preprocessing, feature engineering, model training, and evaluation using different activation functions. The notebook demonstrates how to build a simple neural network (perceptron) from scratch and how to use Scikit-learn's perceptron and MLPClassifier.

## Dataset
The dataset used in this project is the Titanic dataset which contains information about passengers on the Titanic, such as their age, sex, ticket class, and whether they survived the disaster.

### Dataset Columns:
- PassengerId: Unique identifier for each passenger
- Survived: Survival (0 = No, 1 = Yes)
- Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- Name: Passenger name
- Sex: Passenger sex
- Age: Passenger age
- SibSp: Number of siblings/spouses aboard the Titanic
- Parch: Number of parents/children aboard the Titanic
- Ticket: Ticket number
- Fare: Passenger fare
- Cabin: Cabin number
- Embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Steps

### 1. Import Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
```

### 2. Load and Preprocess Data
Load the Titanic dataset and display the first few rows.
```python
data_train = pd.read_csv('tested.csv')
data_train.head(8)
```

### 3. Feature Engineering
Create dictionaries to transform categorical variables into numerical values.
```python
dict_live = {0 : 'Perished', 1 : 'Survived'}
dict_sex = {'male' : 0, 'female' : 1}
data_train['BSex'] = data_train['Sex'].apply(lambda x: dict_sex[x])
```

### 4. Define Features
Select relevant features for the model.
```python
features = data_train[['Pclass', 'BSex']].to_numpy()
print(features)
```

### 5. Activation Functions
Define activation functions like sigmoid and ReLU.
```python
def sigmoid_act(z):
    return 1/(1+ np.exp(-z))

def ReLU_act(z):
    return max(0, z)
```

### 6. Build a Perceptron from Scratch
Implement a simple perceptron model.
```python
def perceptron(X, act):
    np.random.seed(1)
    shapes = X.shape
    n = shapes[0] + shapes[1]
    w = 2*np.random.random(shapes) - 0.5
    w_0 = np.random.random(1)
    f = w_0[0]
    for i in range(0, X.shape[0]-1):
        for j in range(0, X.shape[1]-1):
            f += w[i, j]*X[i,j]/n
    if act == 'Sigmoid':
        output = sigmoid_act(f)
    elif act == "ReLU":
        output = ReLU_act(f)
    return output

print('Output with sigmoid activator: ', perceptron(features, act = 'Sigmoid'))
print('Output with ReLU activator: ', perceptron(features, act = "ReLU"))
```

### 7. Model Training with Scikit-Learn
Load the Iris dataset, standardize features, and train a perceptron model using Scikit-Learn.
```python
# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Train the perceptron
ppn = Perceptron(max_iter=1000, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

# Predictions and accuracy
y_pred = ppn.predict(X_test_std)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
```

### 8. MLP Classifier
Train a Multi-Layer Perceptron classifier and evaluate its performance.
```python
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train_std, y_train)
mlp_y_pred = mlp.predict(X_test_std)
print('Accuracy: %.2f' % accuracy_score(y_test, mlp_y_pred))
```

## Conclusion
This project demonstrates the use of a perceptron and an MLP classifier to predict survival on the Titanic dataset and the Iris dataset. The implementation includes custom perceptron logic and the application of Scikit-Learn's models for classification tasks.

## Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

## How to Run
1. Ensure all required libraries are installed.
2. Place the dataset (`tested.csv`) in the same directory as the notebook.
3. Run each cell in the notebook sequentially.

## Acknowledgments
This project is inspired by the Titanic dataset from Kaggle and the Iris dataset from the UCI Machine Learning Repository.

