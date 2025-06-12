from typing import NoReturn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import os
import linear_regression
import matplotlib.ticker as mtick


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    Preprocess the training set - can change values in the target variables as well
    :param X: data
    :param y: target variable
    :return: clean and processed data
    """
    # Parsing the date and extracting only the year
    X['date'] = pd.to_datetime(X['date'], format='%Y%m%dT000000')
    X['year'] = X['date'].dt.year

    # Change the year renovated to a categorical feature (depending on the year of renovation)
    X['renovated'] = np.where(X['yr_renovated'] == 0, 0, np.where(X['yr_renovated'] <= 2000, 1, 2))

    # Update `yr_renovated` where `yr_built` is greater
    invalid_idx = X[(X['yr_built'] > X['yr_renovated']) & (X['yr_renovated'] != 0)].index
    X.loc[invalid_idx, 'yr_renovated'] = X.loc[invalid_idx, 'yr_built']

    # Dropping unnecessary columns (don't seem to contribute to effectiveness of the model)
    columns_to_drop = ["id", "yr_renovated", "date"]
    X.drop(columns=columns_to_drop, inplace=True)

    # Convert all columns to numeric and handle errors
    X = X.apply(pd.to_numeric, errors='coerce')

    # Align the target variable with the processed features
    y = y.loc[X.index]

    # Fill missing values with column mean
    X = X.apply(lambda col: col.fillna(col.mean()))
    y = y.fillna(y.mean())

    # Drop any remaining rows with NA values
    X = X.dropna()
    # align the target variable
    y = y.loc[X.index]

    return X, y


def preprocess_test(X: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the test data - can't change the target variables (so that the model will be able to fit
    non-cleaned data
    :param X: the data
    :return: processed data
    """
    # Parsing the date and extracting only the year
    X['date'] = pd.to_datetime(X['date'], format='%Y%m%dT000000')
    X['year'] = X['date'].dt.year

    # Change the year renovated to a categorical feature
    X['renovated'] = np.where(X['yr_renovated'] == 0, 0, np.where(X['yr_renovated'] <= 2000, 1, 2))

    # Update `yr_renovated` where `yr_built` is greater
    invalid_idx = X[(X['yr_built'] > X['yr_renovated']) & (X['yr_renovated'] != 0)].index
    X.loc[invalid_idx, 'yr_renovated'] = X.loc[invalid_idx, 'yr_built']

    # Dropping unnecessary columns
    columns_to_drop = ["id", "yr_renovated", "date"]
    X.drop(columns=columns_to_drop, inplace=True)

    # Convert all columns to numeric and handle errors
    X = X.apply(pd.to_numeric, errors='coerce')

    # Fill missing values with column mean
    X = X.apply(lambda col: col.fillna(col.mean()))

    return X


def evaluate_model(X_tr, y_tr, X_te, y_te):
    """
    Fitting the linear regression model over increasing percentages and measuring the loss (over the test set)
    Parameters
    ----------
    X_tr : pd.DataFrame
        X training set
    y_tr : pd.Series
        y training set
    X_te : pd.DataFrame
        X test set
    y_te : pd.Series
        y test set
    Returns
    -------
    None
        A graph representing the connection between the sample size and the loss and confidence.
    """
    # all the percentages to fit over
    percentages = np.arange(10, 101)
    avg_loss_list = []
    var_loss_list = []

    indices = np.arange(len(X_tr))

    # iterating over all the percentages
    for percentage in percentages:
        cur_loss_list = []
        # for each percentage - sampling 10 times
        for _ in range(10):
            np.random.shuffle(indices)
            num_samples = int(len(X_tr) * (percentage / 100.0))
            selected_indices = indices[:num_samples]
            # using the randomly selected indices
            X_subset = X_tr.iloc[selected_indices]
            y_subset = y_tr.iloc[selected_indices]

            cur_model = linear_regression.LinearRegression()
            cur_model.fit(X_subset.values, y_subset.values)

            loss = cur_model.loss(X_te.values, y_te.values)
            cur_loss_list.append(loss)
        # appending the results to the general list
        avg_loss_list.append(np.mean(cur_loss_list))
        var_loss_list.append(np.var(cur_loss_list))

    avg_loss_list = np.array(avg_loss_list)
    std_losses = np.sqrt(var_loss_list)
    # naming labels
    x_label = "Percentage of Training Data"
    y_label = "Mean Squared Error"
    title = "Model Performance vs. Training Size"
    # plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, avg_loss_list, label="Average Loss")
    # inserting std error ribbon
    plt.fill_between(percentages, avg_loss_list - 2 * std_losses, avg_loss_list + 2 * std_losses,
                     color='green', alpha=0.2, label="Error Ribbon (±2 std)")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Ensure full numbers in the y-axis
    plt.gca().yaxis.set_major_locator(mtick.MaxNLocator(integer=True))

    plt.show()


def pearson_corr(a, b):
    """
    Calculates the Pearson correlation coefficient between a and b.

    Parameters
    ----------
    a : np.ndarray
        First variable.
    b : np.ndarray
        Second variable.

    Returns
    -------
    float
        Pearson correlation coefficient between a and b.
    """
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    cov = np.sum((a - mean_a) * (b - mean_b))
    denominator = np.sqrt(np.sum((a - mean_a) ** 2) * np.sum((b - mean_b) ** 2))
    return cov / denominator


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : pd.Series of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    os.makedirs(output_path, exist_ok=True)
    for feature in X.columns:
        corr = pearson_corr(X[feature].values, y.values)

        plt.figure(figsize=(8, 6))
        plt.scatter(X[feature], y, color='blue')
        plt.title(f"Correlation Between {feature} and Price\nPearson Correlation {corr:.2f}")
        plt.xlabel(f"{feature}")
        plt.ylabel("Price")
        plt.grid(True)

        file_path = os.path.join(output_path, f"pearson_correlation_{feature}.png")
        plt.savefig(file_path)
        plt.close()


if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test
    np.random.seed(76)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=76)

    # Question 3 - preprocessing of housing prices train dataset
    X_processed_train, y_processed_train = preprocess_train(X_train, y_train)

    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_processed_train, y_processed_train)

    # Question 5 - preprocess the test data
    X_processed_test = preprocess_test(X_test)
    y_processed_test = y_test.loc[X_processed_test.index]

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    evaluate_model(X_processed_train, y_processed_train, X_processed_test, y_processed_test)
