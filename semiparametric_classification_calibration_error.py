import math

import numpy as np
from sklearn.model_selection import KFold
from scipy.interpolate import UnivariateSpline
import scipy.stats

class SemiparametricMiscalibration(object):
    def __init__(self, folds=5, epsilon = 0.05, weights = 'constant', bootstrap_size = 500, orthogonal=False, normalize=False, smoothing='kernel'):
        self.folds = folds
        self.kf = KFold(n_splits=folds, shuffle=True, random_state=708)
        self.double_fold = KFold(n_splits=2)
        self.epsilon = epsilon
        self.orthogonal = orthogonal
        self.bootstrap_size = bootstrap_size

        self.smoothing = smoothing

        self.normalize = normalize

        if weights is None or weights == 'constant':
            self.weight_function = lambda x: np.ones(x.shape[0])
        elif weights == 'relative':
            self.weight_function = lambda x: self._relative_weights(x)
        elif weights == 'chi':
            self.weight_function = lambda x: self._max_power_weights(x)
        else:
            raise Exception('Weight function "{}" unknown'.format(weights))

    def _relative_weights(self, X):
        return 1.0 / np.maximum(np.minimum(X**2, (1 - X)**2), self.epsilon)

    def _max_power_weights(self, X):
        return 1.0 / np.maximum(X * (1 - X), self.epsilon)

    def _weighted_mean(self, v, weights):
        if self.normalize:
            return np.mean(v * weights) / np.mean(weights)
        else:
            return np.mean(v * weights)

    def _weighted_se(self, v, weights):
        n = v.shape[0]
        out = np.zeros(self.bootstrap_size)
        for b in range(self.bootstrap_size):
            boot_idx = np.random.choice(n, n)
            out[b] = self._weighted_mean(v[boot_idx], weights[boot_idx])
        return np.std(out)

    def _calculate_miscalibration(self, X, Y, a):
        stat = ((X- Y) * (X - a))
        if_ = ((X - a) * (X + a - 2*Y))

        if self.orthogonal:
            est = self._weighted_mean(if_, self.weight_function(X))
        else:
            est = self._weighted_mean(stat, self.weight_function(X))
        se = self._weighted_se(if_, self.weight_function(X))

        return est, se

    def rms_calibration_error_conf_int(self, X, Y, hyperparam_range=None, alpha=0.05):
        est, se = self.calculate_miscalibration_crossfit(X, Y, hyperparam_range=None)
        z_alpha_div_2 = -scipy.stats.norm.ppf(alpha / 2.0)
        return (np.sqrt(max(est - z_alpha_div_2 * se, 0)),
                np.sqrt(max(est, 0)),
                np.sqrt(max(est + z_alpha_div_2 * se, 0)))

    def calculate_miscalibration_crossfit(self, X, Y, hyperparam_range=None):
        if hyperparam_range is None:
            hyperparam_range = np.linspace(1, X.shape[0] / 2.0, 40)
        return self._calculate_miscalibration(
            X, Y,
            self._calculate_opt_cross_fit_calibration_function(X, Y, hyperparam_range))

    def _calculate_calibration_function(self, X_train, Y_train, X_test, sigma=1):
        if self.smoothing == 'kernel':
            dists = np.abs(X_train[np.newaxis,:] - X_test[:,np.newaxis])
            kernel = np.exp(-sigma * (dists ** 2))
            preds = kernel.dot(Y_train) / kernel.sum(axis=1)
        elif self.smoothing == 'spline':
            ord = np.argsort(X_train)
            s = UnivariateSpline(X_train[ord], Y_train[ord], s=sigma)
            preds = s(X_test)
        else:
            raise Exception('Smoothing type "{}" not implemented'.format(self.smoothing))
        return preds

    def _calculate_cross_fit_calibration_function(self, X, Y, hyperparams):
        a = np.zeros(Y.shape)
        for train_index, test_index in self.kf.split(X):
           X_train, X_test = X[train_index], X[test_index]
           Y_train, Y_test = Y[train_index], Y[test_index]
           a[test_index] = self._calculate_calibration_function(X_train, Y_train, X_test, hyperparams)
        return(a)

    def _choose_opt_calibration_hyperparam(self, X, Y, hyperparam_range):
        weights = self.weight_function(X)
        weights /= np.mean(weights)

        best_error = np.float('inf')
        best_hyperparam = None
        for hyperparam in hyperparam_range:
            est = self._calculate_cross_fit_calibration_function(X, Y, hyperparam)
            error = np.mean(weights * (est - Y) ** 2)
            if error < best_error:
                best_error = error
                best_hyperparam = hyperparam
        return best_hyperparam

    def _get_undersmoothed_hyperparam(self, X, Y, hyperparam_range):
        n = Y.shape[0]
        opt_hyperparam = self._choose_opt_calibration_hyperparam(X, Y, hyperparam_range)
        if self.smoothing == 'kernel':
            opt_hyperparam /= n ** 0.08
        else:
            # For now, don't change spline hyperparams, since it's not clear how to choose.
            # Should be slightly undersmoothed, though. Simulations might help elucidate.
            pass
        return opt_hyperparam

    def _calculate_opt_cross_fit_calibration_function(self, X, Y, hyperparam_range):
        hyperparam = self._get_undersmoothed_hyperparam(X, Y, hyperparam_range)
        return self._calculate_cross_fit_calibration_function(X, Y, hyperparam)
