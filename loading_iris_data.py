# Import the library
from sklearn.datasets import load_iris
iris = load_iris()
print("Dataset loaded successfully!")
print(iris.data)
print("Feature names:")
print(iris.feature_names)
print("Flower species:")
print(iris.target_names)
print("Dataset shape:", iris.data.shape)

##using pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("All libraries imported successfully!")
