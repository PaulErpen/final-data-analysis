import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from scipy.integrate import trapezoid as trapz


def logistic_hamid(x, a, b, c, d, e):
    tmp = 0.5 - 1.0 / (1 + np.exp(b * (x - c)))
    y = a * tmp + d + e * x
    return y


def regress_hamid(X, Y):
    """
    Python implementation of the regress_hamid function.
    """
    temp = np.corrcoef(X, Y)[0, 1]  # Extract the correlation coefficient

    if temp > 0:
        beta0 = [
            np.abs(np.max(Y) - np.min(Y)),
            1 / np.std(X),
            np.mean(X),
            np.mean(Y),
            1,
        ]
    else:
        beta0 = [
            -np.abs(np.max(Y) - np.min(Y)),
            1 / np.std(X),
            np.mean(X),
            np.mean(Y),
            1,
        ]

    # Use curve_fit for nonlinear least squares fitting
    popt, pcov = curve_fit(
        logistic_hamid, X, Y, p0=beta0, maxfev=100000, method="dogbox"
    )  # Increase maxfev if needed

    beta = popt  # Extract the fitted parameters
    yhat = logistic_hamid(X, *beta)  # Calculate the predicted values

    ehat = Y - yhat  # Calculate the residuals

    mY = np.mean(Y)
    rsqr = 1 - np.sum(ehat**2) / np.sum((Y - mY) ** 2)  # Calculate R-squared

    return yhat, beta, ehat, rsqr


def opinion_norm(scores, stds):
    """
    Python implementation of the opinion_norm function.
    """
    param = {}  # Create an empty dictionary to store the parameters

    param["w"] = 1 / (np.max(scores) - np.min(scores))
    param["b"] = -np.min(scores) / (np.max(scores) - np.min(scores))
    stds = stds * param["w"] * 100
    param["th_min"] = 2 * np.min(stds)
    param["th_max"] = 2 * np.max(stds)

    return param


def PWRC(pred, label, th, param):
    """
    Python implementation of the PWRC function.
    """
    # Check if inputs are column vectors and transpose if necessary
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)  # or pred = pred[:, np.newaxis]
    if label.ndim == 1:
        label = label.reshape(-1, 1)  # or label = label[:, np.newaxis]

    # MOS/DMOS normalization
    if param["flag"]:  # Note the dictionary access
        label = (param["w"] * label + param["b"]) * 100
        pred = (param["w"] * pred + param["b"]) * 100
    else:
        label = (1 - param["w"] * label - param["b"]) * 100
        pred = (1 - param["w"] * pred - param["b"]) * 100

    # Locate range index
    min_idx = np.argmin(np.abs(th - param["th_min"]))
    max_idx = np.argmin(np.abs(th - param["th_max"]))

    c = 0.175

    label_r = (
        np.argsort(np.argsort(label.flatten())) + 1
    )  # Double argsort for ranking, +1 for 1-based indexing
    pred_r = (
        np.argsort(np.argsort(pred.flatten())) + 1
    )  # Double argsort for ranking, +1 for 1-based indexing

    len_ = len(label)
    len_short = 2 * (len_ - 1)

    # Pre-computing for importance weight w
    w = []  # Use a list to store the weights
    omega = 0
    for i in range(len_ - 1):
        level = (
            np.max(
                np.stack([label_r[: len_ - i - 1], label_r[i + 1 :]], axis=-1), axis=1
            )
            - 1
        )  # Fixed indexing
        diff = np.abs(label_r[: len_ - i - 1] - pred_r[: len_ - i - 1]) + np.abs(
            label_r[i + 1 :] - pred_r[i + 1 :]
        )
        w_i = np.exp(level / (len_ - 1) + diff / len_short)
        w.append(w_i)  # Append to the list
        omega = omega + np.sum(w_i)

    for i in range(len_ - 1):
        w[i] = w[i] / omega

    if param["act"]:
        er_rate = np.zeros(len(th))
        for step in range(len(th)):
            for i in range(len_ - 1):
                temp_pred = pred[: len_ - i - 1] - pred[i + 1 :]
                temp_label = label[: len_ - i - 1] - label[i + 1 :]
                active_label = 1.0 / (1 + np.exp(-c * (np.abs(temp_label) - th[step])))
                er_label = np.sign(temp_pred) * np.sign(temp_label)
                idx = er_label * active_label
                er_rate[step] = er_rate[step] + np.sum(w[i] * idx)

        AUC = trapz(
            er_rate[min_idx : max_idx + 1], th[min_idx : max_idx + 1]
        )  # Corrected indexing for trapz
    else:
        er_rate = 0
        for i in range(len_ - 1):
            temp_pred = pred[: len_ - i - 1] - pred[i + 1 :]
            temp_label = label[: len_ - i - 1] - label[i + 1 :]
            er_label = np.sign(temp_pred) * np.sign(temp_label)
            er_rate = er_rate + np.sum(w[i] * er_label)

        AUC = er_rate

    return er_rate, AUC


def delta_MOS(pred, dmos, p):
    """
    Placeholder for delta_MOS function. Needs implementation.
    """
    raise NotImplementedError("delta_MOS function not yet implemented.")


def compute_pwrc(LIVE_SSIM, dmos):
    pred, beta, ehat, rsqr = regress_hamid(LIVE_SSIM, dmos)

    dmos_std = np.std(dmos)

    # parameter preparation
    p = opinion_norm(dmos, dmos_std)
    p["flag"] = 0  # DMOS -> flag 0; MOS -> flag 1;
    p["act"] = 1  # enable/disable A(x,T): p.act->1/0
    th = np.arange(0, 111, 0.5)  # Use np.arange for the range

    PWRC_th, AUC = PWRC(pred, dmos, th, p)

    return AUC
    # print(f"The AUC value of SSIM is {AUC}")


#
# delta_value = delta_MOS(pred, dmos, p)
# print(f"The delta MOS value of SSIM is {delta_value}")
#
# # SA-ST curve
# plt.figure()
# plt.plot(th, PWRC_th, color='r', linewidth=1.5, linestyle='-')
# plt.xlabel('T') # Italic not directly supported in standard matplotlib, but you can use LaTeX if needed
# plt.ylabel('PWRC')
# plt.legend(['SSIM'])
# plt.grid(True)
# plt.title('SA-ST curve on LIVE II database') # Italic not directly supported in standard matplotlib.
# plt.show()
