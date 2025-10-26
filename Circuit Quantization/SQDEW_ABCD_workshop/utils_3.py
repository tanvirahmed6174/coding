# utils_3.py  — Coupled resonator fitting (magnitude + complex)
from __future__ import annotations
from typing import Optional, Tuple, Union

import numpy as np
import lmfit

# ---------------------------- Units ---------------------------------
Hz  = 1.0
kHz = 1e3 * Hz
MHz = 1e6 * Hz
GHz = 1e9 * Hz
pi = np.pi

# Scale factor: GHz <-> rad/s
w_scale = 2 * pi * GHz  # multiply GHz -> rad/s ; divide rad/s -> GHz


# --------------------- Helpers: dip finding / FWHM -------------------
def _get_w0_inds_and_masks(ws: np.ndarray,
                           s21s: np.ndarray,
                           nr_dips: int = 2):
    """Find up to `nr_dips` minima in |S21| after linear background removal."""
    w0_ind_list = []
    w0_mask_list = []

    a = np.mean(np.abs(s21s)[[0, -1]])
    b = np.diff(np.abs(s21s)[[0, -1]])[0] / np.diff(ws[[0, -1]])[0]
    threshold_vals = a * 0.85 + b * (ws - ws.mean())
    s_abs_tilted = np.abs(s21s).copy() - threshold_vals

    for i in range(nr_dips):
        w0_mask_list.append([True] * len(ws))
        s_masked = s_abs_tilted[np.logical_and.reduce(w0_mask_list)].copy()
        w0_ind_list.append(np.argmin(np.abs(s_abs_tilted - np.min(s_masked))))

        # grow mask left
        for j in range(w0_ind_list[i], -1, -1):
            if s_abs_tilted[j] < 0:
                w0_mask_list[i][j] = False
            else:
                break
        # grow mask right
        for j in range(w0_ind_list[i], len(s_abs_tilted)):
            if s_abs_tilted[j] < 0:
                w0_mask_list[i][j] = False
            else:
                break

    w0_ind_list, w0_mask_list = zip(*sorted(zip(w0_ind_list, w0_mask_list)))
    return w0_ind_list, w0_mask_list


def _get_fwhm(ws: np.ndarray,
              s21s_abs_tilted_shifted: np.ndarray,
              kappa_threshold: np.ndarray,
              w0_ind_list,
              w0_mask_list,
              i: int):
    """Estimate FWHM around the i-th dip by crossing the kappa_threshold."""
    w0 = ws[w0_ind_list[i]]

    m_left = (~np.array(w0_mask_list[i])) & (ws < w0)
    if not np.any(m_left):
        fwhm_left = w0
    else:
        idx = np.argmin(np.abs(np.abs(s21s_abs_tilted_shifted[m_left]) - kappa_threshold[m_left]))
        fwhm_left = ws[m_left][idx]

    m_right = (~np.array(w0_mask_list[i])) & (ws > w0)
    if not np.any(m_right):
        fwhm_right = w0
    else:
        idx = np.argmin(np.abs(np.abs(s21s_abs_tilted_shifted[m_right]) - kappa_threshold[m_right]))
        fwhm_right = ws[m_right][idx]

    return (fwhm_right - fwhm_left)


# ---------------------- Coupled-mode: guess --------------------------
def graphical_guess_coupled(ws_GHz: np.ndarray, s21s: np.ndarray, nr_dips: int = 2):
    """
    Graphical guess for coupled (filter+resonator) in **GHz** space.
    Returns A, k, phi, kappa_p, omega_p, omega_r, J (GHz units).
    """
    a = np.mean(np.abs(s21s)[[0, -1]])
    b = np.diff(np.abs(s21s)[[0, -1]])[0] / np.diff(ws_GHz[[0, -1]])[0]
    threshold_vals = a + b * (ws_GHz - ws_GHz.mean())

    s_abs_tilted = np.abs(s21s).copy() - threshold_vals
    s_abs_tilted_shifted = s_abs_tilted + np.abs(np.min(s_abs_tilted))

    a_ts = np.mean(s_abs_tilted_shifted)
    thresh_ts = np.full_like(ws_GHz, a_ts)
    kappa_threshold = np.sqrt(0.5) * thresh_ts

    max_ts = np.max(np.abs(s_abs_tilted_shifted))
    r_norm = np.abs(s_abs_tilted_shifted[-1]) / max_ts
    l_norm = np.abs(s_abs_tilted_shifted[0]) / max_ts
    phi_abs = np.arccos((r_norm + l_norm) / 2)
    kappa_threshold *= np.abs(np.cos(phi_abs))

    try:
        w0_inds, w0_masks = _get_w0_inds_and_masks(ws_GHz, s_abs_tilted_shifted, nr_dips=nr_dips)
    except Exception:
        w0_inds, w0_masks = _get_w0_inds_and_masks(ws_GHz, s_abs_tilted_shifted, nr_dips=1)
        nr_dips = 1

    fwhms = [_get_fwhm(ws_GHz, s_abs_tilted_shifted, kappa_threshold, w0_inds, w0_masks, i)
             for i in range(nr_dips)]

    if nr_dips == 2:
        w1, w2 = sorted([ws_GHz[idx] for idx in w0_inds[:2]])
        wr = 0.5 * (w1 + w2)
        wp = 0.5 * (w1 + w2)
        J = 0.5 * np.abs(w1 - w2)
        kappa_p = np.sum(fwhms)
    else:
        wr = ws_GHz[w0_inds[0]]
        wp = wr
        J = fwhms[0] / 10
        kappa_p = fwhms[0]

    overall_fwhm = (np.max(ws_GHz) - np.min(ws_GHz)) if nr_dips == 1 else (np.abs(w2 - w1) + np.max(fwhms))
    m = np.abs(ws_GHz - wr) < overall_fwhm / 2
    phi_sign = np.sign(s_abs_tilted_shifted[m][-1] - s_abs_tilted_shifted[m][0]) if np.any(m) else 1.0
    phi = phi_sign * phi_abs

    return dict(A=a, k=b, phi=phi, kappa_p=kappa_p, omega_p=wp, omega_r=wr, J=J)


# -------------- Models (GHz space) ----------------------------------
def coupled_hanger_reparam_mag(w, A, k, phi, kappa_p, omega_c, delta, J):
    """
    Magnitude model (returns |S|). Reparam with omega_c=(ωp+ωr)/2, delta=(ωp-ωr)/2.
    """
    omega_0 = np.mean(w)
    omega_p = omega_c + delta
    omega_r = omega_c - delta

    Delta_p = w - omega_p
    Delta_r = w - omega_r

    numer = kappa_p * (-2j * Delta_p)
    denom = 4 * J**2 + (kappa_p - 2j * Delta_p) * (-2j * Delta_r)
    frac = numer / denom

    s = np.cos(phi) - np.exp(1j * phi) * frac
    s = (A + k * (w - omega_0)) * s
    return np.abs(s)


def coupled_hanger_complex(w, a0_re, a0_im, a1_re, a1_im, phi, kappa_p, omega_p, omega_r, J):
    """Complex model: (a0 + a1*(w-w0)) * coupled_response(w). Returns complex S21."""
    a0 = a0_re + 1j*a0_im
    a1 = a1_re + 1j*a1_im
    w0 = np.mean(w)
    Δp, Δr = w-omega_p, w-omega_r
    numer = kappa_p*(-2j*Δp)
    denom = 4*J**2 + (kappa_p - 2j*Δp)*(-2j*Δr)
    frac = numer/denom
    s = np.cos(phi) - np.exp(1j*phi)*frac
    bg = a0 + a1*(w - w0)
    return bg*s


# ------------------ Weights for dip-focused refit --------------------
def _weights_two_dips(w, w1, w2, width):
    g1 = np.exp(-0.5 * ((w - w1) / max(width, 1e-9))**2)
    g2 = np.exp(-0.5 * ((w - w2) / max(width, 1e-9))**2)
    return 1.0 + 4.0 * (g1 + g2)


# -------------------- Magnitude-only fitter --------------------------
def fit_resonance_coupled(
    ws: Optional[np.ndarray] = None,
    s21s: Optional[np.ndarray] = None,
    manual_graph_guess: Optional[dict] = None,
    return_full: bool = False,
    verbose: bool = True
) -> Union[Tuple[dict, np.ndarray], Tuple[lmfit.model.ModelResult, dict, np.ndarray]]:
    """
    Fit the coupled (filter+resonator) **magnitude** model. INPUT ws in rad/s.
    RETURNS:
      if return_full=False:
         (fit_results_dict_in_rad/s, s21_fit_trace_mag_on_ws_grid)
      if return_full=True:
         (ModelResult_in_GHz_space_or_None, fit_results_dict_in_rad/s, s21_fit_trace_mag)
    """
    ws_GHz = ws / w_scale
    graph = graphical_guess_coupled(ws_GHz, s21s)
    if manual_graph_guess is not None:
        graph.update(**manual_graph_guess)

    omega_c0 = 0.5 * (graph['omega_p'] + graph['omega_r'])
    delta0   = abs(0.5 * (graph['omega_p'] - graph['omega_r']))

    model = lmfit.Model(coupled_hanger_reparam_mag)
    params = model.make_params(
        A=graph['A'], k=graph['k'], phi=float(np.clip(graph['phi'], -np.pi, np.pi)),
        kappa_p=max(graph['kappa_p'], 1e-9),
        omega_c=omega_c0, delta=max(delta0, 0.0),
        J=max(graph['J'], 1e-9)
    )
    params['kappa_p'].min = 0.0
    params['J'].min       = 0.0
    params['delta'].min   = 0.0
    params['phi'].min     = -np.pi
    params['phi'].max     =  np.pi

    # Stage 1 (broad)
    overall_span = 50 * graph['kappa_p']
    center = 0.5 * (graph['omega_p'] + graph['omega_r'])
    m1 = np.abs(ws_GHz - center) < overall_span

    result1 = None
    try:
        result1 = model.fit(
            data=np.abs(s21s[m1]),
            w=ws_GHz[m1],
            params=params,
            method="least_squares",
            fit_kws=dict(loss="soft_l1", f_scale=0.1),
            nan_policy="omit",
        )
        best = result1.best_values
    except Exception as e:
        if verbose: print("Stage-1 fit failed:", e)
        best = dict(A=graph['A'], k=graph['k'], phi=graph['phi'],
                    kappa_p=graph['kappa_p'], omega_c=omega_c0,
                    delta=delta0, J=graph['J'])

    # Stage 2 (tight + weighted)
    omega_c1 = best.get('omega_c', omega_c0)
    delta1   = abs(best.get('delta', delta0))
    w1 = omega_c1 - delta1
    w2 = omega_c1 + delta1
    width = max(best.get('kappa_p', graph['kappa_p']), graph['kappa_p'])

    m2 = (np.abs(ws_GHz - w1) < 10 * width) | (np.abs(ws_GHz - w2) < 10 * width)
    w_fit = ws_GHz[m2]
    y_fit = np.abs(s21s[m2])
    weights = _weights_two_dips(w_fit, w1, w2, width)

    params2 = model.make_params(**best)
    params2['kappa_p'].min = 0.0
    params2['J'].min       = 0.0
    params2['delta'].min   = 0.0
    params2['phi'].min     = -np.pi
    params2['phi'].max     =  np.pi

    result = None
    try:
        result = model.fit(
            data=y_fit,
            w=w_fit,
            params=params2,
            method="least_squares",
            fit_kws=dict(loss="soft_l1", f_scale=0.05),
            weights=weights,
            nan_policy="omit",
        )
        best = result.best_values
        if verbose: print(result.fit_report())
    except Exception as e:
        if verbose: print("Stage-2 fit failed:", e)

    # Choose best solution
    if result is None and result1 is not None:
        best = result1.best_values
    elif result is None and result1 is None:
        best = dict(A=graph['A'], k=graph['k'], phi=graph['phi'],
                    kappa_p=graph['kappa_p'], omega_c=omega_c0,
                    delta=delta0, J=graph['J'])

    # Map back to (omega_p, omega_r) in GHz space
    omega_p = best['omega_c'] + abs(best['delta'])
    omega_r = best['omega_c'] - abs(best['delta'])

    # Build outputs
    fit_results_dict = dict(
        A=best['A'], k=best['k'], phi=best['phi'],
        kappa_p=best['kappa_p'] * w_scale,
        omega_p=omega_p         * w_scale,
        omega_r=omega_r         * w_scale,
        J=best['J']             * w_scale,
    )
    # Fitted magnitude on full grid
    s21_fit_mag = coupled_hanger_reparam_mag(ws_GHz, **best)

    return (result if result is not None else result1, fit_results_dict, s21_fit_mag) if return_full \
           else (fit_results_dict, s21_fit_mag)


# -------------------- Complex-valued fitter --------------------------
def fit_resonance_coupled_complex(
    ws: Optional[np.ndarray] = None,
    s21s: Optional[np.ndarray] = None,
    manual_graph_guess: Optional[dict] = None,
    return_full: bool = False,
    verbose: bool = True
) -> Union[Tuple[dict, np.ndarray], Tuple[lmfit.model.ModelResult, dict, np.ndarray]]:
    """
    Fit the coupled model to **complex S21** with a complex affine background.
    INPUT ws in rad/s.
    RETURNS:
      if return_full=False:
         (fit_results_dict_in_rad/s, |s21_fit_complex| on ws grid)
      if return_full=True:
         (ModelResult_in_GHz_space, fit_results_dict_in_rad/s, |s21_fit_complex|)
    """
    ws_GHz = ws / w_scale
    graph = graphical_guess_coupled(ws_GHz, s21s)
    if manual_graph_guess is not None:
        graph.update(**manual_graph_guess)

    # Initial complex background from endpoints
    a0 = s21s[0]
    params = lmfit.Parameters()
    params.add('a0_re', value=np.real(a0))
    params.add('a0_im', value=np.imag(a0))
    params.add('a1_re', value=0.0)
    params.add('a1_im', value=0.0)
    params.add('phi',    value=float(np.clip(graph['phi'], -np.pi, np.pi)), min=-np.pi, max=np.pi)
    params.add('kappa_p',value=max(graph['kappa_p'], 1e-9), min=0)
    params.add('omega_p',value=graph['omega_p'], min=0)
    params.add('omega_r',value=graph['omega_r'], min=0)
    params.add('J',      value=max(graph['J'], 1e-9), min=0)

    # Residual = stack of real/imag parts
    def _resid(pars):
        y = coupled_hanger_complex(ws_GHz, **pars.valuesdict())
        d = s21s
        return np.column_stack((np.real(y)-np.real(d), np.imag(y)-np.imag(d))).ravel()

    mini = lmfit.Minimizer(_resid, params, nan_policy='omit')
    result = mini.least_squares(method='trf', loss='soft_l1', f_scale=0.05)
    best = result.params.valuesdict()

    # Derived quantities (GHz)
    omega_p = best['omega_p']
    omega_r = best['omega_r']
    # Build outputs (rad/s)
    fit_results_dict = dict(
        phi=best['phi'],
        kappa_p=best['kappa_p'] * w_scale,
        omega_p=omega_p         * w_scale,
        omega_r=omega_r         * w_scale,
        J=best['J']             * w_scale,
        # background (complex) is left in GHz-space params only (a0/a1),
        # but the returned fit trace is |S21| on ws grid:
    )

    s21_fit_complex = coupled_hanger_complex(ws_GHz, **best)
    s21_fit_mag = np.abs(s21_fit_complex)

    if verbose:
        print(result.fit_report())

    return (result, fit_results_dict, s21_fit_mag) if return_full \
           else (fit_results_dict, s21_fit_mag)
