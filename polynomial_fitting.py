from typing import NoReturn
import linear_regression
from linear_regression import LinearRegression
import numpy as np


class PolynomialFitting(LinearRegression):
    """
    Polynomial Fitting using Least Squares estimation
    """

    def __init__(self, k: int):
        """
        Instantiate a polynomial fitting estimator

        Parameters
        ----------
        k : int
            Degree of polynomial to fit
        """
        # initializing using inheritance - super
        super().__init__(include_intercept=True)  # Always include intercept for polynomial fitting
        self.k = k
        self.linear_model = linear_regression.LinearRegression(False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to polynomial transformed samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        X_poly = self.__transform(X)  # transforming to vandermonde matrix
        self.linear_model.fit(X_poly, y)  # using linear regression func for fitting

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
        return self.linear_model.predict(X)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

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
        X_poly = self.__transform(X)  # Transforming to Vandermonde matrix
        return self.linear_model.loss(X_poly, y)  # Using linear regression loss method

    def __transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform given input according to the univariate polynomial transformation

        Parameters
        ----------
        X: ndarray of shape (n_samples,)

        Returns
        -------
        transformed: ndarray of shape (n_samples, k+1)
            Vandermonde matrix of given samples up to degree k
        """
        X_flat = X.flatten()  # flattening in order to use np.vander func - receives a flattened matrix
        poly_degree = self.k + 1  # in order to have x to the power of 0
        return np.vander(X_flat, poly_degree, True)
