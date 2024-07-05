import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import time

def load_data():
    df = pd.read_csv("Data/Raw/HousingData.csv")
    df = df.drop(columns = ["CRIM", "ZN", "CHAS", "RAD", "B", "AGE"])
    df["INDUS"] = df["INDUS"].fillna(df["INDUS"].mean())
    df["LSTAT"] = df["LSTAT"].fillna(df["LSTAT"].mean())
    return df

def headmap(df):
    plt.figure(figsize=(20, 10))
    heatmap = sns.heatmap(df.corr().abs(), annot=True)
    plt.title("Heatmap of Variable Importance")
    fig = heatmap.get_figure()
    return fig

def normalize(input, scaler):
    x_test_norm = scaler.transform(input)
    return x_test_norm

def rfml_pred(model, input):
    pred = model.predict(input)
    return pred

def predict(model, scaler, INDUS, NOX, RM, DIS, TAX, PTRATIO, LSTAT):
    columns = ['INDUS', 'NOX', 'RM', 'DIS', 'TAX', 'PTRATIO', 'LSTAT']
    row = np.array([INDUS, NOX, RM, DIS, TAX, PTRATIO, LSTAT])
    X = pd.DataFrame([row], columns=columns)
    norm_data = normalize(X, scaler)
    pred = rfml_pred(model, norm_data)
    return pred

def plot_correlations_with_target(df, target_column):
    correlations = df.corr()[target_column].drop(target_column)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=correlations.index, y=correlations.values)
    plt.xticks(rotation=90)
    plt.title(f"Correlation of All Columns with {target_column}")
    plt.xlabel("Columns")
    plt.ylabel("Correlation")
    fig = plt.gcf()
    return fig
