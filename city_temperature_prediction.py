import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from polynomial_fitting import PolynomialFitting
from sklearn.model_selection import train_test_split
import numpy as np


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # Load dataset
    df_to_return = pd.read_csv(filename)
    # parsing the date column
    df_to_return['Date'] = pd.to_datetime(df_to_return['Date'])
    # Drop rows with any missing values
    df_to_return.dropna(inplace=True)
    # Add 'DayOfYear' column
    df_to_return['DayOfYear'] = df_to_return['Date'].dt.dayofyear
    # Filter the DataFrame to include only rows where 'Temp' is between -55 and 45 inclusive
    # (note that lowest temp recorded is -89 and highest is 56)
    df_to_return = df_to_return[(df_to_return['Temp'] >= -55) & (df_to_return['Temp'] <= 45)]
    return df_to_return


def question_3(data_set: pd.DataFrame):
    """
    Filters the dataset to contain only samples from Israel and investigates how the average daily temp changes as a
    function of the day of the year.
    :param data_set: the dataset
    """
    ############# FIRST-PART #############
    israel_df = data_set[data_set['Country'] == 'Israel']
    # Plot scatter plot of Temp vs. DayOfYear
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=israel_df, x='DayOfYear', y='Temp', hue=israel_df['Date'].dt.year, palette='tab10')
    # naming labels
    title = "Average Daily Temperature in Israel by Day of Year"
    x_label = "Day of Year"
    y_label = "Temperature (°C)"
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title='Year')
    plt.show()

    ############# SECOND-PART #############
    # Group by Month and calculate standard deviation
    israel_df.loc[:, 'Month'] = israel_df['Date'].dt.month
    month_std = israel_df.groupby('Month')['Temp'].agg('std').reset_index()

    # Plot bar plot of standard deviation by month
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(month_std)))

    # naming labels
    title = "Standard Deviation of Daily Temperatures by Month in Israel"
    x_label = "Month"
    y_label = "Temperature Standard Deviation (°C)"
    plt.bar(month_std['Month'], month_std['Temp'], color=colors)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def question_4(data_set: pd.DataFrame):
    """
    Group the samples according to 'Country' and 'Month', calculate the average and standard deviation of the
     temperature, and plot a line plot of the average monthly temperature with error bars.
    Parameters
    ----------
    data_set : pd.DataFrame
        The city temperature dataset
    """
    # Add 'Month' column
    data_set['Month'] = data_set['Date'].dt.month

    # Group by 'Country' and 'Month' and calculate mean and std of 'Temp'
    grouped = data_set.groupby(['Country', 'Month']).agg(avg_temp=('Temp', 'mean'),
                                                         std_temp=('Temp', 'std')).reset_index()

    # Plot the data using matplotlib
    plt.figure(figsize=(12, 8))

    countries = grouped['Country'].unique()
    for country in countries:
        country_data = grouped[grouped['Country'] == country]
        plt.errorbar(
            country_data['Month'],
            country_data['avg_temp'],
            yerr=country_data['std_temp'],
            label=country,
            capsize=5
        )
    # naming labels
    title = "Average Monthly Temperature with Standard Deviation by Country"
    x_label = "Month"
    y_label = "Average Temperature (°C)"

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title='Country')
    plt.grid(True)
    plt.show()


def question_5(data_set: pd.DataFrame):
    """
    Fitting polynomial models for different values of `k` and evaluating the model performance.

    Parameters
    ----------
    data_set : pd.DataFrame
        The city temperature dataset
    """
    israel_df = data_set[data_set['Country'] == 'Israel']
    X = israel_df['DayOfYear'].values.reshape(-1, 1)
    y = israel_df['Temp'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluate polynomial models with degrees 1 through 10
    degrees = list(range(1, 11))
    train_errors = []
    test_errors = []

    # iterating over degrees
    for k in degrees:
        poly_model = PolynomialFitting(k)
        poly_model.fit(X_train, y_train)
        train_loss = poly_model.loss(X_train, y_train)
        test_loss = poly_model.loss(X_test, y_test)
        train_errors.append(train_loss)
        test_errors.append(test_loss)

        # Print the test error for the current degree
        print(f'Degree {k}: Test Error = {test_loss}')

    # Identify the best degree (simplest model with the lowest test error)
    best_degree = degrees[test_errors.index(min(test_errors))]
    print(f'\nBest degree: {best_degree}')

    # Plot bar plot of test errors for different polynomial degrees
    plt.figure(figsize=(10, 6))
    plt.bar(degrees, test_errors, color='blue')

    # naming labels
    title = "Test Error by Polynomial Degree"
    x_label = "Polynomial Degree"
    y_label = "Mean Squared Error"

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()


def question_6(data_set: pd.DataFrame):
    """
    Fit a model using k=5 over the entire subset of records from Israel.
    Plot a bar plot showing the model's error over each of the other countries.

    Parameters
    ----------
    data_set : pd.DataFrame
        The city temperature dataset
    """
    # Fit model for Israel with k=5
    israel_df = data_set[data_set['Country'] == 'Israel']
    X_israel = israel_df['DayOfYear'].values.reshape(-1, 1)
    y_israel = israel_df['Temp'].values

    poly_model = PolynomialFitting(5)
    poly_model.fit(X_israel, y_israel)

    # Calculate the model's error for each other country
    errors = {}
    for country in data_set['Country'].unique():
        if country != 'Israel':
            country_df = data_set[data_set['Country'] == country]
            X_country = country_df['DayOfYear'].values.reshape(-1, 1)
            y_country = country_df['Temp'].values

            test_loss = poly_model.loss(X_country, y_country)
            errors[country] = test_loss

    # Convert dict keys and values to lists for plotting
    countries = list(errors.keys())
    mse_values = list(errors.values())

    # pick colors
    colors = ['pink', 'red', 'yellow']
    # Plot bar plot of model's error for each country
    plt.figure(figsize=(10, 6))
    plt.bar(countries, mse_values, color=colors)

    # naming labels
    title = "Model Error for Each Country (k=5, fitted on Israel)"
    x_label = "Country"
    y_label = "Mean Squared Error"

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")
    # Question 3 - Exploring data for specific country
    question_3(df)
    # Question 4 - Exploring differences between countries
    question_4(df)
    # Question 5 - Fitting model for different values of `k`
    question_5(df)
    # Question 6 - Evaluating fitted model on different countries
    question_6(df)
