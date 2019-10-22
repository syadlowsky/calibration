# calibration
Code implementing "A Calibration Metric for Risk Scores with Survival Data" (MLHC 2019)

# Python
The main implementation is in Python. See `semiparametric_calibration_error.py`. With an object constructed with the appropriate hyperparameters, call the `calculate_miscalibration_crossfit` with the `X` the predicted probability, `Y` the study time outcome, and `D` a binary variable indicating `1` for an event or `0` if the observation is censored.

# R
This calls the Python implementation using `reticulate`. Assumes you are calling this from the one directory up (so that this repo can be submodule'd into another project). The import on line 4 can be adjusted for different usage patterns. Use `semiparametric_censored_miscalibration` to construct a Python object and pass this object, as well as the data using the same signature as above to `calculate_miscalibration` to compute the calibration error.
