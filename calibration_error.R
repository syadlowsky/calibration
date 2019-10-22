#install.packages("reticulate")

library(reticulate)
py_calibration = import("calibration.semiparametric_calibration_error")
np = import("numpy")

semiparametric_censored_miscalibration = function(folds=5, epsilon = 0.05, t_0 = 10, survival_aggregation="Kaplan-Meier", kernel_fn="rbf", verbose=FALSE) {
    obj = list()
    obj$py = py_calibration$SemiparametricCensoredMiscalibration(as.integer(folds), epsilon, t_0, survival_aggregation, kernel_fn, verbose)
    class(obj) = "semiparametric_censored_miscalibration"
    return(obj)
}

calculate_miscalibration = function(obj, prediction, time, event) {
    results = obj$py$calculate_miscalibration_crossfit(np$array(prediction), np$array(time), np$array(event))
    return(list(error = results[1], se = results[2]))
}
