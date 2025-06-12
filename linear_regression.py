import numpy as np
from typing import NoReturn


class LinearRegression:

    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem

    Attributes
    ----------
    fitted_ : bool
        Indicates if estimator has been fitted. Set to True in ``self.fit`` function

    include_intercept_: bool
        Should fitted model include an intercept or not

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by linear regression. To be set in
        `LinearRegression.fit` function.
    """

    def __init__(self, include_intercept: bool = True):
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not
        """
        self.include_intercept = include_intercept
        self.coefs_ = None
        self.fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if self.include_intercept:
            # Add a column of ones to X for the intercept term
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        # Solve the normal equation to find the best fit coefficients
        self.coefs_ = np.linalg.pinv(X) @ y
        self.fitted_ = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # checking that the model has been fitted and that coefs are valid
        if not self.fitted_:
            raise ValueError("The model has not been fitted yet. Please fit the model before predicting.")

        if self.coefs_ is None:
            raise ValueError("The coefficients are not available. Ensure the model has been properly fitted.")

        if self.include_intercept:
            # Add a column of ones to X for the intercept term
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        return X @ self.coefs_

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under **mean squared error (MSE) loss function**

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        # checking that the model has been fitted
        if not self.fitted_:
            raise ValueError("The model has not been fitted yet. Please fit the model before predicting.")
        y_predicted = self.predict(X)
        mse = np.mean((y_predicted - y) ** 2)
        return float(mse)
