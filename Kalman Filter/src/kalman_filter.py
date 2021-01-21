# -*- coding: utf-8 -*-
import numpy as np


def kalman_transit_covariance(S, A, R):
    """
    :param S: Current covariance matrix
    :param A: Either transition matrix or jacobian matrix
    :param R: Current noise covariance matrix
    """
    new_S = A @ S @ A.T + R
    return new_S


def kalman_process_observation(mu, S, observation, C, Q):
    """
    Performs processing of an observation coming from the model: z = C * x + noise
    :param mu: Current mean
    :param S: Current covariance matrix
    :param observation: Vector z
    :param C: Observation matrix
    :param Q: Noise covariance matrix (with zero mean)
    """
    new_mu = None
    new_S = None
    K = S @ C.T @ np.linalg.inv(C @ S @ C.T + Q)
    new_mu = mu + K @ (observation - C @ mu)
    new_S = (np.eye(S.shape[0]) - K @ C) @ S
    return new_mu, new_S
