# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 10:48:22 2025

@author: Tanvir
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from lmfit import minimize, Parameters, report_fit

# === Load Data ===
QA = np.loadtxt("C:/_Data/datas/20250908/015317_hist_sweep.txt")
X = QA[0:41, :]
Y = QA[41:82, :]
Z = QA[82:123, :]

x = X[0]
y = Y[:, 0]

# Interpolation function for smooth Z
Z = np.nan_to_num(Z, nan=0.0)
interp_Z = RegularGridInterpolator((y, x), Z, bounds_error=False, fill_value=0.0)

# Flattened coordinate grid
xx, yy = np.meshgrid(x, y)
flat_points = np.stack([yy.ravel(), xx.ravel()], axis=1)
Z_vals = interp_Z(flat_points)

# === One Function for lmfit ===
def objective(params):
    x0 = params['x0']
    y0 = params['y0']
    r = 5000      # fixed radius
    k = 0.05     # smoothness of the edge

    # Distance from each point to the center
    dx = flat_points[:, 1] - x0
    dy = flat_points[:, 0] - y0
    dist = np.sqrt(dx**2 + dy**2)

    # Smooth circular mask (sigmoid edge)
    mask = 1 / (1 + np.exp(k * (dist - r)))

    # Negative of Z under mask (we're minimizing, so this maximizes Z inside)
    return -Z_vals * mask

# === Initialize Parameters ===
params = Parameters()
params.add('x0', value=(x[0] + x[-1]) / 2)
params.add('y0', value=(y[0] + y[-1]) / 2)

# === Fit ===
result = minimize(objective, params, method='least_squares')
report_fit(result)

# === Extract Results ===
x_opt = result.params['x0'].value
y_opt = result.params['y0'].value
x_err = result.params['x0'].stderr
y_err = result.params['y0'].stderr

# === Plot ===
plt.figure(figsize=(6, 5))
plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
plt.colorbar(label='Z')
circle = plt.Circle((x_opt, y_opt), 5000, color='red', fill=False, linewidth=2)
plt.gca().add_patch(circle)
plt.plot(x_opt, y_opt, 'r+', markersize=10)
plt.title("Fitted Circle Maximizing Z")
plt.xlabel("I")
plt.ylabel("Q")
plt.tight_layout()
plt.show()

# === Print Result ===
print(f"Fitted center: x = {x_opt:.2f} ± {x_err:.2f}, y = {y_opt:.2f} ± {y_err:.2f}")
