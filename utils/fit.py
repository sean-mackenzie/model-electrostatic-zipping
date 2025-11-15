import math
import os
import pandas as pd
import numpy as np
from scipy.interpolate import splrep, splev


def fit_smoothing_spline(x, y, s, xnew):
    """

    :param x:
    :param y:
    :param s: smoothing factor, where higher values equal more smoothing.
    :return:
    """
    tck = splrep(x, y, s=s)
    ynew = splev(xnew, tck, der=0)
    return ynew

def wrapper_fit_radial_membrane_profile(x, y, s, faux_r_zero=None, faux_r_edge=None):
    """

    :param x:
    :param y:
    :param s:
    :param faux_r_zero: y_faux = np.mean(y[x < faux_r_zero]) (e.g., faux_r_zero = dict_settings['radius_hole_microns'])
    :param faux_r_edge: x_faux = faux_r_edge (e.g., faux_r_edge = dict_settings['radius_microns'])
    :return:
    """
    if faux_r_zero is not None:
        x_fake, y_fake = 0, np.mean(y[x < faux_r_zero])

        x_fake = np.linspace(x_fake, faux_r_zero, num=5, endpoint=False)
        y_fake = np.repeat(y_fake, 5)

        x = np.append(x_fake, x)
        y = np.append(y_fake, y)
    if faux_r_edge is not None:
        x_fake, y_fake = faux_r_edge, 0
        x = np.append(x, x_fake)
        y = np.append(y, y_fake)

    # sort values (in case faux_r_edge is not largest radial position)
    sort_indices = np.argsort(x)
    # Sort both arrays using the sorted indices
    x = x[sort_indices]
    y = y[sort_indices]

    xnew = np.linspace(x.min(), x.max(), 50)
    ynew = fit_smoothing_spline(x, y, s, xnew)

    return xnew, ynew, x, y

def wrapper_fit_radial_membrane_profile_(x, y, s, dict_settings, faux_r_zero, faux_r_edge):
    if faux_r_zero:
        x_fake, y_fake = 0, np.mean(y[x < dict_settings['radius_hole_microns']])
        x = np.append(x_fake, x)
        y = np.append(y_fake, y)
    if faux_r_edge:
        x_fake, y_fake = dict_settings['radius_microns'], 0
        x = np.append(x, x_fake)
        y = np.append(y, y_fake)

    xnew = np.linspace(x.min(), x.max(), 200)
    ynew = fit_smoothing_spline(x, y, s, xnew)

    return xnew, ynew, x, y


def fit_power_law_scaling(x, y):
    """
    Fit a power-law scaling model in log-log space and compute model statistics.

    This function takes in two arrays, `x` and `y`, performs a power-law scaling
    fit of the form :math:`V = a \cdot E^x` via ordinary least squares (OLS) in
    the log-log space, and computes relevant statistics including fit coefficients,
    confidence intervals, residuals, and the coefficient of determination (R²) in
    log-log space. Additionally, a summary dictionary of the results is generated.

    The parameters `x` and `y` must both be provided, with `x` representing the
    independent variable and `y` the dependent variable. Logarithms are computed on
    both inputs for fitting purposes, and the result includes all information summarized
    in a dictionary returned as a pandas DataFrame.

    :param x: Array of input values representing the independent variable used
              in the power-law regression. Values should be positive and numeric.
    :param y: Array of observed output values representing the dependent variable
              corresponding to `x`. Values should also be positive and numeric.
    :return: A pandas DataFrame containing the summary of the power-law fit,
             including estimated coefficients, confidence intervals, and R² in
             log-log space.
    """
    n_all = len(x)
    n_used = len(y)
    # --- Fit power law in log-log space: ln(V) = ln(a) + x * ln(E) ---
    x_log = np.log(x)
    y_log = np.log(y)

    # Design matrix for OLS
    X = np.column_stack([np.ones_like(x_log), x_log])  # [1, ln(E)]
    # OLS closed-form: beta = (X^T X)^{-1} X^T y
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y_log)
    b0, b1 = beta  # b0 = ln(a), b1 = exponent x

    # Predictions and residuals
    yhat = X @ beta
    res = y_log - yhat
    RSS = float(np.sum(res**2))
    TSS = float(np.sum((y_log - np.mean(y_log))**2))
    R2_log = 1.0 - RSS / TSS if TSS > 0 else float("nan")

    # --- Standard errors & 95% CI (Student t; fallback to normal if needed) ---
    dfree = max(n_used - 2, 1)  # degrees of freedom for 2-parameter model
    sigma2 = RSS / dfree if n_used > 2 else float("nan")
    cov = sigma2 * XtX_inv
    se_b0 = math.sqrt(cov[0, 0]) if np.isfinite(cov[0, 0]) else float("nan")
    se_b1 = math.sqrt(cov[1, 1]) if np.isfinite(cov[1, 1]) else float("nan")

    # Try to get t critical; if SciPy not available, use 1.96
    try:
        from scipy.stats import t as student_t  # type: ignore
        tcrit = float(student_t.ppf(0.975, dfree))
    except Exception:
        tcrit = 1.96

    a_hat = float(np.exp(b0))  # prefactor (units: V / MPa^x)
    x_hat = float(b1)          # exponent (unitless)

    a_ci = (math.exp(b0 - tcrit * se_b0), math.exp(b0 + tcrit * se_b0)) if np.isfinite(se_b0) else (float("nan"), float("nan"))
    x_ci = (x_hat - tcrit * se_b1, x_hat + tcrit * se_b1) if np.isfinite(se_b1) else (float("nan"), float("nan"))

    # --- Prepare a neat text summary ---
    summary = {
        "n_rows_in_file": n_all,
        "n_used_for_fit": n_used,
        "model": "V_PI = a * (E_MPa)^x",
        "prefactor_a_units": "V / (MPa^x)",
        "a_hat": a_hat,
        "a_95CI_low": a_ci[0],
        "a_95CI_high": a_ci[1],
        "exponent_x_hat": x_hat,
        "x_95CI_low": x_ci[0],
        "x_95CI_high": x_ci[1],
        "R2_in_log_space": R2_log,
    }

    return summary

def fit_pre_stretch_scaling(x, y):
    lam = x
    V = y

    def ols_fit(X, y):
        X_ = np.column_stack([np.ones(len(X)), X])
        XtX = X_.T @ X_
        beta = np.linalg.inv(XtX) @ (X_.T @ y)
        yhat = X_ @ beta
        res = y - yhat
        RSS = float(np.sum(res ** 2))
        TSS = float(np.sum((y - np.mean(y)) ** 2))
        R2 = 1.0 - RSS / TSS if TSS > 0 else float("nan")
        dfree = max(len(y) - X_.shape[1], 1)
        sigma2 = RSS / dfree if len(y) > X_.shape[1] else float("nan")
        cov = sigma2 * np.linalg.inv(XtX)
        se = np.sqrt(np.diag(cov))
        try:
            from scipy.stats import t as student_t  # type: ignore
            tcrit = float(student_t.ppf(0.975, dfree))
        except Exception:
            tcrit = 1.96
        ci = np.column_stack([beta - tcrit * se, beta + tcrit * se])
        return beta, se, ci, R2, yhat

    # Power-law
    xlog = np.log(lam)
    ylog = np.log(V)
    beta_pl, se_pl, ci_pl, R2_pl, _ = ols_fit(xlog, ylog)
    a_hat = float(np.exp(beta_pl[0]));
    x_hat = float(beta_pl[1])
    a_ci = (math.exp(ci_pl[0, 0]), math.exp(ci_pl[0, 1]))
    x_ci = (ci_pl[1, 0], ci_pl[1, 1])

    # V^2 ~ (1 - λ^-4)
    s1 = 1.0 - lam ** (-4.0)
    beta_s1, se_s1, ci_s1, R2_s1, _ = ols_fit(s1, V ** 2)

    # V^2 ~ (λ^2 - λ^-2)
    s2 = lam ** 2 - lam ** (-2.0)
    beta_s2, se_s2, ci_s2, R2_s2, _ = ols_fit(s2, V ** 2)

    # V^2 ~ (1 - λ^-6)
    s3 = 1.0 - lam ** (-6.0)
    beta_s3, se_s3, ci_s3, R2_s3, _ = ols_fit(s3, V ** 2)

    def fmt(v):
        return f"{v:.6g}"

    print("Power-law (log–log): V = a * λ^x")
    print("  a_hat =", fmt(a_hat), " (95% CI:", fmt(a_ci[0]), ",", fmt(a_ci[1]) + ")")
    print("  x_hat =", fmt(x_hat), " (95% CI:", fmt(x_ci[0]), ",", fmt(x_ci[1]) + ")")
    print("  R^2   =", fmt(R2_pl))
    print()

    print("Neo-Hookean tension model: V^2 = c0 + c1*(1 - λ^-4)")
    print("  c0 =", fmt(beta_s1[0]), " (95% CI:", fmt(ci_s1[0, 0]), ",", fmt(ci_s1[0, 1]) + ")")
    print("  c1 =", fmt(beta_s1[1]), " (95% CI:", fmt(ci_s1[1, 0]), ",", fmt(ci_s1[1, 1]) + ")")
    print("  R^2 =", fmt(R2_s1))
    print("  Baseline V at λ=1 (from c0):", fmt(np.sqrt(beta_s1[0])), "V")
    print()

    print("Alt. invariant: V^2 = c0 + c1*(λ^2 - λ^-2)")
    print("  c0 =", fmt(beta_s2[0]), " (95% CI:", fmt(ci_s2[0, 0]), ",", fmt(ci_s2[0, 1]) + ")")
    print("  c1 =", fmt(beta_s2[1]), " (95% CI:", fmt(ci_s2[1, 0]), ",", fmt(ci_s2[1, 1]) + ")")
    print("  R^2 =", fmt(R2_s2))
    print()

    print("------- BELOW IS THE PHYSICALLY CORRECT FORM ------------ ")
    print("Neo-Hookean tension model: V^2 = c0 + c1*(1 - λ^-6)")
    print("  c0 =", fmt(beta_s3[0]), " (95% CI:", fmt(ci_s3[0, 0]), ",", fmt(ci_s3[0, 1]) + ")")
    print("  c1 =", fmt(beta_s3[1]), " (95% CI:", fmt(ci_s3[1, 0]), ",", fmt(ci_s3[1, 1]) + ")")
    print("  R^2 =", fmt(R2_s3))
    print("  Baseline V at λ=1 (from c0):", fmt(np.sqrt(beta_s3[0])), "V")

    return beta_s3[0], beta_s3[1]

def neo_hookean_tension_model(lam, c0, c1):
    """
    The Neo-Hookean tension model is: V^2 = c0 + c1*(1 - λ^-6)
    So, to plot V vs. function, use: V = np.sqrt(c0 + c1*(1 - λ^-6))

    :param lam: Deformation stretch ratio.
    :param c0: Material parameter, often related to the modulus.
    :param c1: Material parameter, related to additional effects in the
               model.
    :return: Computed tension value based on the input parameters.
    :rtype: float
    """
    return np.sqrt(c0 + c1 * (1 - lam ** (-6)))

# -