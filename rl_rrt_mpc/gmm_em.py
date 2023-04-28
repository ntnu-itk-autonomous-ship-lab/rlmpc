"""
    gmm_em.py

    Summary:
        Contains a class for a Gaussian Mixture Model with the Expectation-Maximization algorithm.

    Author: https://github.com/mr-easy/GMM-EM-Python, Trym Tengesdal
"""

import random

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal


class GMM_EM:
    def __init__(self, k, dim, init_mu=None, init_sigma=None, init_pi=None, colors=None) -> None:
        """
        Define a model with known number of clusters and dimensions.
        input:
            - k: Number of Gaussian clusters
            - dim: Dimension
            - init_mu: initial value of mean of clusters (k, dim)
                       (default) random from uniform[-10, 10]
            - init_sigma: initial value of covariance matrix of clusters (k, dim, dim)
                          (default) Identity matrix for each cluster
            - init_pi: initial value of cluster weights (k,)
                       (default) equal value to all cluster i.e. 1/k
            - colors: Color valu for plotting each cluster (k, 3)
                      (default) random from uniform[0, 1]
        """
        self.k = k
        self.dim = dim
        if init_mu is None:
            init_mu = random.rand(k, dim) * 20 - 10
        self.mu = init_mu
        if init_sigma is None:
            init_sigma = np.zeros((k, dim, dim))
            for i in range(k):
                init_sigma[i] = np.eye(dim)
        self.sigma = init_sigma
        if init_pi is None:
            init_pi = np.ones(self.k) / self.k
        self.pi = init_pi
        if colors is None:
            colors = random.rand(k, 3)
        self.colors = colors

    def run(self, num_iters: int) -> np.ndarray:
        """Run EM algorithm for a number of iterations.

        Args:
            num_iters (int): Number of iterations to run EM algorithm.

        Returns:
            np.ndarray: Cluster means.
        """
        num_iters = 30
        log_likelihood = [self.log_likelihood(self.data)]
        self.plot("Iteration:  0")
        for e in range(num_iters):
            self.e_step()
            self.m_step()
            log_likelihood.append(self.log_likelihood(self.data))
            print(f"Iteration: {e + 1}, log-likelihood: {log_likelihood[-1]}")
            self.plot(title="Iteration: " + str(e + 1))
        return self.mu, self.sigma, self.pi

    def init_em(self, X: np.ndarray) -> None:
        """
        Initialization for EM algorithm.
        input:
            - X: data (batch_size, dim)
        """
        self.data = X
        self.num_points = X.shape[0]
        self.z = np.zeros((self.num_points, self.k))

    def e_step(self):
        """
        E-step of EM algorithm.
        """
        for i in range(self.k):
            self.z[:, i] = self.pi[i] * multivariate_normal.pdf(self.data, mean=self.mu[i], cov=self.sigma[i])
        self.z /= self.z.sum(axis=1, keepdims=True)

    def m_step(self):
        """
        M-step of EM algorithm.
        """
        sum_z = self.z.sum(axis=0)
        self.pi = sum_z / self.num_points
        self.mu = np.matmul(self.z.T, self.data)
        self.mu /= sum_z[:, None]
        for i in range(self.k):
            j = np.expand_dims(self.data, axis=1) - self.mu[i]
            s = np.matmul(j.transpose([0, 2, 1]), j)
            self.sigma[i] = np.matmul(s.transpose(1, 2, 0), self.z[:, i])
            self.sigma[i] /= sum_z[i]

    def log_likelihood(self, X: np.ndarray) -> float:
        """
        Compute the log-likelihood of X under current parameters
        input:
            - X: Data (batch_size, dim)
        output:
            - log-likelihood of X: Sum_n Sum_k log(pi_k * N( X_n | mu_k, sigma_k ))
        """
        ll = []
        for d in X:
            tot = 0
            for i in range(self.k):
                tot += self.pi[i] * multivariate_normal.pdf(d, mean=self.mu[i], cov=self.sigma[i])
            ll.append(np.log(tot))
        return np.sum(ll)

    def plot(self, title: str) -> None:
        """
        Draw the data points and the fitted mixture model.
        input:
            - title: title of plot and name with which it will be saved.
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.scatter(self.data[:, 0], self.data[:, 1], s=3, alpha=0.4)
        ax.scatter(self.mu[:, 0], self.mu[:, 1], c=self.colors)
        self.draw(ax, lw=3)
        ax.set_xlim((-12, 12))
        ax.set_ylim((-12, 12))

        plt.title(title)
        plt.savefig(title.replace(":", "_"))
        plt.show()
        plt.clf()

    def plot_gaussian(self, mean: np.ndarray, cov: np.ndarray, ax: plt.Axes, n_std: float = 3.0, facecolor: str = "none", **kwargs) -> plt.Axes:
        """
        Utility function to plot one Gaussian from mean and covariance.
        """
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = mean[0]
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = mean[1]
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def draw(self, ax: plt.Axes, n_std: float = 2.0, facecolor: str = "none", **kwargs) -> None:
        """
        Function to draw the Gaussians.
        Note: Only for two-dimensionl dataset
        """
        if self.dim != 2:
            print("Drawing available only for 2D case.")
            return
        for i in range(self.k):
            self.plot_gaussian(self.mu[i], self.sigma[i], ax, n_std=n_std, edgecolor=self.colors[i], **kwargs)
