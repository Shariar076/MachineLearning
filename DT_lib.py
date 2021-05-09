import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
import scipy.stats as sps

# Load in the data and define the column labels
dataset = pd.read_csv('adult.csv')
dataset = dataset.sample(frac=1)

# Encode the feature values from strings to integers since the sklearn DecisionTreeClassifier only takes numerical values
for label in dataset.columns:
    dataset[label] = LabelEncoder().fit(dataset[label]).transform(dataset[label])

Tree_model = DecisionTreeClassifier(criterion="entropy", max_depth=1)
array = dataset.values
Y = array[:, -1]
X = array[:, :-1]

cv = cross_validate(Tree_model, X, Y, cv= 5)
# print cv
predictions = np.mean(cv['test_score'])
print('The accuracy is: ', predictions * 100, '%')