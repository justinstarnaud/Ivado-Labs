import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import re
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

def remove_square_brackets(value):
    return re.sub(r'\[.*?\]', '', value)

def remove_parentheses(value):
    return re.sub(r'[<\(].*?[>\)]', '', value)

def convert_million(value):
    value = value.strip()
    million_match = re.match(r'^([\d.]+)\s*million$', value)
    if million_match:
        return str(int(float(million_match.group(1)) * 1000000))
    return value
    
def convert_to_float(value):
    value = value.strip()
    comma_match = re.match(r'^([\d,]+)$', value)
    if comma_match:
        return float(comma_match.group(1).replace(',', ''))

def select_table_from_html(url):
    res = requests.get(url)
    soup = bs(res.text, 'html.parser')
    table = soup.find_all('table')[0]
    html_str = str(table)
    return pd.read_html(StringIO(html_str))[0]

def split_data(X, y):
    return train_test_split(X, y, test_size=0.1, random_state=42)

def linear_regression(X,y):
    param_grid = {
        'fit_intercept': [True, False],
        'copy_X' : [True, False]
    }
    return train_view_model(X, y, LinearRegression(), param_grid)

def knn_regressor(X, y):
    param_grid = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'weights': ['uniform', 'distance']}
    return train_view_model(X, y, KNeighborsRegressor(), param_grid, False)

def decision_tree_regressor(X, y):
    param_grid = {
        'criterion': ['squared_error', 'poisson', 'friedman_mse', 'absolute_error'],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    return train_view_model(X, y, DecisionTreeRegressor(), param_grid, False)


def train_view_model(X, y, model, param_grid, plot=True):
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Cross validation with 5 fold and mean squared error
    grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f'Best Parameters: {best_params}')

    best_model = grid_search.best_estimator_

    test_predictions = best_model.predict(X_test)

    mse = mean_squared_error(y_test, test_predictions)
    print(f'Mean Squared Error on Test Set: {mse}')

    title = f'{type(model).__name__} results on test set'
    view_data(X_test, y_test, title, test_predictions, plot)

    return best_model

def predict_view_model(X, y, model, plot=True):
    predictions = model.predict(X)
    title = f'{type(model).__name__} museum visitors predictions'
    view_data(X, y, title, predictions, plot)

def view_data(X,y, title, predictions=None, plot=False):
    plt.scatter(X, y, color='blue', label='True data')
    if predictions is not None:
        plt.scatter(X, predictions, color='red', label='Predictions')
        if plot: 
            plt.plot(X, predictions, color='green', linewidth=2, label='Regression')
    plt.title(title)
    plt.xlabel('Population')
    plt.ylabel('Amount of visitor')
    plt.legend()
    plt.show()






