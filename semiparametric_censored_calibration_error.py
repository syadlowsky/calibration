import math

import numpy as np
from sklearn.model_selection import KFold
from scipy.interpolate import UnivariateSpline

class SemiparametricCensoredMiscalibration(object):
    def __init__(self, folds=5, epsilon = 0.05, t_0 = 10, survival_aggregation="Kaplan-Meier", kernel_fn="rbf", verbose=False):
        self.folds = folds
        self.kf = KFold(n_splits=folds, shuffle=True)
        self.epsilon = epsilon
        self.t_0 = t_0
        self.survival_aggregation = survival_aggregation
        self.verbose=verbose
        self.kernel_fn = kernel_fn

    def _calculate_miscalibration(self, X, Y, D, dLambda, S_C, cond_mean, compensator, H_2):
        n = X.shape[0]
        times = self._get_times(Y)
        a = cond_mean[:,0]

        scaling = 1/np.maximum(np.minimum(X ** 2, (1 - X) ** 2), self.epsilon ** 2)

        # Influence function of IPCW
        at_risk = np.array([y > times for y in Y])
        at_risk_inclusive = np.array([y >= times for y in Y])
        C_process = at_risk * (1-D)[:,np.newaxis]
        dN_C = np.hstack((np.zeros(n)[:,np.newaxis],
                          -np.diff(C_process)))
        
        dM_C = dN_C - dLambda * at_risk
        S_C_minus = S_C
        S_C_minus[:,1:] = S_C[:,0:-1]

        cond_mean[at_risk == False] = 0
        compensator[at_risk == False] = 0

        martingale_integral = (dM_C * (X[:,np.newaxis] - cond_mean) * (X - a)[:,np.newaxis] / np.maximum(S_C_minus, 0.001)).sum(axis=1)

        # Influence function of recalibration function
        Y_loc = np.where(Y[:,np.newaxis] == times[np.newaxis,:])
        km_influence_denom = np.ones(n)
        km_influence_denom[Y_loc[0]] = 1 / H_2[Y_loc]
        s = np.sum(compensator[:,:-1] * at_risk[:,:-1], axis=1)
        compensator = (a-X)* (1-a) * (D * (Y <= self.t_0) * km_influence_denom - s)

        # IPCW
        matches = np.where(np.logical_and((D==1)[:,np.newaxis], Y[:,np.newaxis] == times[np.newaxis,:]))
        ipw_weights = np.zeros(n)
        ipw_weights[Y > self.t_0] = 1 / S_C[Y > self.t_0, -2]
        ipw_weights[matches[0]] = 1 / S_C[matches]
        if self.verbose:
            print(ipw_weights.max())
        ipw_weights = np.minimum(ipw_weights, 20)

        stat = ipw_weights * (X - (Y <= self.t_0)) * (X - a)

        if_ = (ipw_weights * ((a - X) * ((Y <= self.t_0) - X)) + compensator - martingale_integral)
        if_mean = np.mean(if_ * scaling)
        return np.mean(stat * scaling), math.sqrt(np.mean(scaling*(if_ - if_mean)**2) / Y.shape[0])

    def calculate_miscalibration_crossfit(self, X, Y, D):
        def fitter():
            C_cum_hazard_increments, S_C, cond_mean, compensator, H_2, _ = \
                self._calculate_cross_fit_calibration_function_opt(
                    X, Y, D)
            return self._calculate_miscalibration(
                X, Y, D, C_cum_hazard_increments, S_C, cond_mean, compensator, H_2)
        many_runs = np.array([fitter() for i in range(5)])
        return np.nanmedian(many_runs, axis=0)

    def _get_times(self, Y):
        t_0 = self.t_0
        epsilon = 0
        if np.any(Y > t_0):
            epsilon = np.min(Y[Y > t_0]) - t_0
        times = np.unique(Y)
        times = times[times <= t_0]
        if times[0] != 0:
            times = np.insert(times, 0, 0)
        if times[-1] != t_0:
            times = np.append(times, t_0)

        times = np.append(times, t_0 + epsilon / 2)
        return times

    def _calculate_cross_fit_calibration_function_opt(self, X, Y, D):
        n = X.shape[0]
        results = None
        error = float('inf')
        opt_sigma = 0
        for sigma in np.exp(np.arange(-4.5, 0.4, 0.2)):
            results_sigma = self._calculate_cross_fit_calibration_function_for_hyperparam(
                X, Y, D, sigma)
            if self.verbose:
                print(sigma, error, results_sigma[-1], results_sigma[-1].sum())
            if error > results_sigma[-1].sum():
                results = results_sigma
                opt_sigma = sigma
                error = results_sigma[-1].sum()

        return self._calculate_cross_fit_calibration_function_for_hyperparam(
                X, Y, D, opt_sigma / (n**0.08))

    def _calculate_cross_fit_calibration_function_for_hyperparam(self, X, Y, D, bandwidth=30):
        times = self._get_times(Y)
        n_times = times.shape[0]
        n = Y.shape[0]

        C_cum_hazard_increments = np.zeros((n, n_times))
        S_C = np.ones((n, n_times))
        cond_mean = np.zeros((n, n_times))
        compensator = np.zeros((n, n_times))
        H_2_complete = np.zeros((n, n_times))

        error = np.array([0,0]).astype(np.double)
        epsilon = self.epsilon
        for train_index, test_index in self.kf.split(X):
           X_train, X_test = X[train_index], X[test_index]
           Y_train, Y_test = Y[train_index], Y[test_index]
           D_train, D_test = D[train_index], D[test_index]
           test_Cch, test_S_C, test_cond_mean, H_T, H_C, H_2, \
              test_compensator = self._calculate_calibration_function(X_train, Y_train, D_train, times, X_test, bandwidth)
           C_cum_hazard_increments[test_index,:] = test_Cch
           S_C[test_index, :] = test_S_C
           cond_mean[test_index, :] = test_cond_mean
           compensator[test_index, :] = test_compensator
           H_2_complete[test_index, :] = H_2

           at_risk = np.array([y > times for y in Y_test])
           T_process = at_risk * D_test[:,np.newaxis]
           C_process = at_risk * (1-D_test)[:,np.newaxis]

           scaling = 1/np.maximum(np.minimum(X_test**2,(1-X_test)**2), epsilon**2)
           scaling = scaling[:, np.newaxis]

           error += np.array([(scaling * (T_process - H_T) ** 2).mean(), (scaling * (C_process - H_C) ** 2).mean()])

        return (C_cum_hazard_increments, S_C, cond_mean, compensator, H_2_complete, error)

    def _calculate_calibration_function(self, X_train, Y_train, D_train, times, X_test, sigma=1):
        at_risk = np.array([y > times for y in Y_train])
        T_process = at_risk * D_train[:,np.newaxis]
        C_process = at_risk * (1-D_train)[:,np.newaxis]

        dists = np.abs(X_train[np.newaxis,:] - X_test[:,np.newaxis]) / sigma
        if self.kernel_fn == "cubic":
            kernel = 15*(1-(dists**2))**3/16
            kernel[dists > 1] = 0
        elif self.kernel_fn == "rbf":
            kernel = np.exp(-dists**2)
        else:
            raise("Kernel {} not supported".format(self.kernel_fn))

        H_T = kernel.dot(T_process) / kernel.sum(axis=1)[:,np.newaxis]
        H_C = kernel.dot(C_process) / kernel.sum(axis=1)[:,np.newaxis]
        H_2 = kernel.dot(at_risk) / kernel.sum(axis=1)[:,np.newaxis]

        dH_T = -np.insert(np.diff(H_T, axis=1), 0, 0, axis=1)
        dH_C = -np.insert(np.diff(H_C, axis=1), 0, 0, axis=1)

        H_2_minus = np.insert(H_2, 0, 1, axis=1)[:,:-1]

        T_cum_hazard_increments = dH_T / H_2_minus
        C_cum_hazard_increments = dH_C / H_2_minus
        T_cum_hazard_increments[dH_T==0] = 0
        C_cum_hazard_increments[dH_C==0] = 0

        if self.survival_aggregation == "Nelson-Aalen":
            S_T = np.exp(np.cumsum(-T_cum_hazard_increments, axis=1))
            S_C = np.exp(np.cumsum(-C_cum_hazard_increments, axis=1))
        else:
            S_T = np.cumprod(1-T_cum_hazard_increments, axis=1)
            S_C = np.cumprod(1-C_cum_hazard_increments, axis=1)

        compensator = dH_T / (H_2_minus*(H_2_minus+dH_T))

        # Assumes last time step is self.t_0 + \epsilon
        cond_mean = 1-(S_T[:,-2][:, np.newaxis] / S_T)

        return (C_cum_hazard_increments, S_C, cond_mean, H_T, H_C, H_2, compensator)
