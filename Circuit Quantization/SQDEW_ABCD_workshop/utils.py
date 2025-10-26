import lmfit 
import numpy as np
from typing import *

Hz = 1; kHz = 1e3*Hz; MHz = 1e6*Hz; GHz = 1e9*Hz
pF = 1e-12; fF = 1e-15; aF = 1e-18
nH = 1e-9
us = 1e-6
ns = 1e-9
um = 1e-6
mm = 1e-3
Ohm = 1
pi = np.pi


w_scale = 2*pi*GHz

def get_w0_inds_and_masks(ws, s21s, nr_dips=2):
    """
    returns list of w0_inds and list of w0_masks
    w0_masks are boolean arrays of length len(ws) with False for points that are in the dip and True otherwise
    w0_inds are the indices of the dips
    w0_inds and w0_masks are in order of increasing w0
    """
    w0_ind_list = []
    w0_mask_list = []

    # get a and b for linear threshold, set the offset to 0.95*a, to increase chances of cutting apart the dips
    a = np.mean(np.abs(s21s)[[-1,0]])
    b = np.diff(np.abs(s21s)[[-1,0]])[0]/np.diff(ws[[-1,0]])[0]
    threshold_vals = a*0.85 + b*(ws - ws.mean())
    s21s_abs_tilted = np.abs(s21s).copy() - threshold_vals

    for i in range(nr_dips):
        w0_mask_list.append([True] * len(ws))
        s21s_abs_tilted_masked = s21s_abs_tilted[np.logical_and.reduce(w0_mask_list)].copy() # copy of s21s with previous dips removed
        w0_ind_list.append(np.argmin(np.abs(s21s_abs_tilted - min(s21s_abs_tilted_masked))))
        # look left from dip, range goes to 0
        for j in range(w0_ind_list[i],-1,-1):
            if s21s_abs_tilted[j] < 0:
                w0_mask_list[i][j] = False
            else:
                break
        # look right from dip, range goes to len-1
        for j in range(w0_ind_list[i],len(s21s_abs_tilted)):
            if s21s_abs_tilted[j] < 0:
                w0_mask_list[i][j] = False
            else:
                break
    
    # sort dips
    w0_ind_list, w0_mask_list = zip(*sorted(zip(w0_ind_list, w0_mask_list)))
    return w0_ind_list, w0_mask_list

def graphical_guess(ws, s21s):

    # use plateau of s21 as reference for kappa
    a = np.mean(np.abs(s21s)[[-1,0]]) # this might be more accurate than np.mean(np.abs(s21s)), especially if phi is large
    b = np.diff(np.abs(s21s)[[-1,0]])[0]/np.diff(ws[[-1,0]])[0]
    threshold_vals = a + b*(ws - ws.mean())
    
    # calulate phi
    # correct s21 for slope, tilt it such that it is flat
    s21s_abs_tilted = np.abs(s21s).copy() - threshold_vals
    s21s_abs_tilted_shifted = s21s_abs_tilted + np.abs(np.min(s21s_abs_tilted)) # shift it such that minimum is at 0

    a_tilted_shifted = np.mean(s21s_abs_tilted_shifted)
    threshold_vals_tilted_shifted = np.array([a_tilted_shifted] * len(ws)) # flat line at the threshold value of the tilted shifted s21

    kappa_threshold = np.sqrt(1/2)*threshold_vals_tilted_shifted # np.sqrt(1/2) of threshold to extract kappa later

    # extract phi from tilted shifted s21
    max_s21s_abs_tilted_shifted = np.max(np.abs(s21s_abs_tilted_shifted))
    right_val_normed = np.abs(s21s_abs_tilted_shifted[-1])/max_s21s_abs_tilted_shifted
    left_val_normed = np.abs(s21s_abs_tilted_shifted[0])/max_s21s_abs_tilted_shifted
    average_val_normed = (right_val_normed + left_val_normed)/2
    phi_abs = np.arccos(average_val_normed)

    kappa_threshold *= np.abs(np.cos(phi_abs)) # lower kappa_threshold by the effect of phi -> if phi is pi/2, kappa_threshold is 0

    def get_fwhm(i):
        # calculate half max
        w0 = ws[w0_ind_list[i]]

        # find ws value to the left of w0 where s21 is (max_s21-wr_s21)/2
        m = ~np.array(w0_mask_list[i]) & (ws < w0)
        if len(ws[m]) == 0:
            fwhm_left = w0
        else:
            fwhm_left = ws[m][np.argmin(np.abs(np.abs(s21s_abs_tilted_shifted[m])-kappa_threshold[m]))]

        # find ws value to the right of w0 where s21 is (max_s21-w1_s21)/2
        m = ~np.array(w0_mask_list[i]) & (ws > w0)
        if len(ws[m]) == 0:
            fwhm_right = w0
        else:
            fwhm_right = ws[m][np.argmin(np.abs(np.abs(s21s_abs_tilted_shifted[m])-kappa_threshold[m]))]

        fwhm = (fwhm_right-fwhm_left)
        return fwhm


    w0_ind_list, w0_mask_list = get_w0_inds_and_masks(ws, s21s_abs_tilted_shifted, nr_dips=1)

    wr = ws[w0_ind_list[0]]

    fwhm = get_fwhm(0)

    m = np.abs(ws - wr) < fwhm
    s21s_abs_tilted_shifted_last = s21s_abs_tilted_shifted[m][-1]
    s21s_abs_tilted_shifted_first = s21s_abs_tilted_shifted[m][0]
    phi_sign = np.sign(s21s_abs_tilted_shifted_last - s21s_abs_tilted_shifted_first)
    phi = phi_sign*phi_abs


    snr = a/np.abs(s21s[w0_ind_list[0]])

    gr = fwhm/snr

    kr = fwhm - gr

    return {
                'wr': wr,
                'kr': kr,
                'gr': gr,
                'phi': phi,
                'a': a,
                'b': b
            }

def hanger_lorentzian(w, wr, kr, gr, phi, a, b, **kw):
            s = np.exp(1j*phi)*kr
            s = s/(2j*(w-wr) - kr - gr)
            s = np.abs(np.cos(phi) + s)
            s = (a + b*(w-w.mean()))*s
            return s


def fit_resonance(
          ws: Optional[np.ndarray]=None, 
          s21s: Optional[np.ndarray]=None, 
          manual_graph_guess: Optional[dict]=None) -> Tuple[dict, np.ndarray]:


        graph_guess = graphical_guess(ws/w_scale, s21s) # type: ignore
        if manual_graph_guess is not None: 
            graph_guess.update(**manual_graph_guess)

        #fit function
        model = lmfit.Model(hanger_lorentzian)
        params = model.make_params(
            wr=graph_guess['wr'],
            kr=graph_guess['kr'],
            phi=graph_guess['phi'],
            gr=graph_guess['gr'],
            a=graph_guess['a'],
            b=graph_guess['b']
        )

        # Set lower bound of 0 for multiple parameters
        for param_name, param in params.items():
            if param_name in ['wr', 'kr']:
                param.min = 0
        params['gr'].set(value=0, vary=False)

        m = np.abs(ws/w_scale-graph_guess['wr']) < 50*graph_guess['kr']

        fit_results_dict = dict()

        try:
            result = model.fit(w=ws[m]/w_scale, data=np.abs(s21s[m]), params=params)

            fit_results_dict.update(**result.best_values)

        except ValueError as e:
            print(f'Fit failed.')
            print('Using graph guess instead.')

            fit_results_dict.update(**graph_guess)
        
        except TypeError as e:
            print(f'Fit failed.')
            print('Using graph guess instead.')

            fit_results_dict.update(**graph_guess)

        s21_fit_trace = hanger_lorentzian(ws/w_scale, **fit_results_dict)

        fit_results_dict['wr'] *= w_scale
        fit_results_dict['kr'] *= w_scale

        return fit_results_dict, s21_fit_trace


# def graphical_guess_coupled(ws, s21s, nr_dips=2):
#     """
#     Graphical guess for coupled mode parameters, assuming hybridized case with nr_dips.
#     Falls back to fewer dips if needed, but optimized for 2.
#     """
#     # Reuse single-mode logic for baseline and tilt
#     a = np.mean(np.abs(s21s)[[-1, 0]])
#     b = np.diff(np.abs(s21s)[[-1, 0]])[0] / np.diff(ws[[-1, 0]])[0]
#     threshold_vals = a + b * (ws - ws.mean())
    
#     s21s_abs_tilted = np.abs(s21s).copy() - threshold_vals
#     s21s_abs_tilted_shifted = s21s_abs_tilted + np.abs(np.min(s21s_abs_tilted))
    
#     a_tilted_shifted = np.mean(s21s_abs_tilted_shifted)
#     threshold_vals_tilted_shifted = np.array([a_tilted_shifted] * len(ws))
    
#     kappa_threshold = np.sqrt(1/2) * threshold_vals_tilted_shifted
    
#     # Estimate phi
#     max_s21s_abs_tilted_shifted = np.max(np.abs(s21s_abs_tilted_shifted))
#     right_val_normed = np.abs(s21s_abs_tilted_shifted[-1]) / max_s21s_abs_tilted_shifted
#     left_val_normed = np.abs(s21s_abs_tilted_shifted[0]) / max_s21s_abs_tilted_shifted
#     average_val_normed = (right_val_normed + left_val_normed) / 2
#     phi_abs = np.arccos(average_val_normed)
    
#     kappa_threshold *= np.abs(np.cos(phi_abs))
    
#     # Get dip indices and masks (try for nr_dips, but handle if fewer)
#     try:
#         w0_ind_list, w0_mask_list = get_w0_inds_and_masks(ws, s21s_abs_tilted_shifted, nr_dips=nr_dips)
#     except:
#         # If fails (e.g., only one dip), fall back to nr_dips=1 and use single-like guesses
#         w0_ind_list, w0_mask_list = get_w0_inds_and_masks(ws, s21s_abs_tilted_shifted, nr_dips=1)
#         nr_dips = 1
    
#     # Compute FWHMs
#     fwhms = [get_fwhm(i) for i in range(nr_dips)]
    
#     if nr_dips == 2:
#         w1, w2 = sorted([ws[w0_ind_list[i]] for i in range(2)])
#         wr = (w1 + w2) / 2
#         wp = (w1 + w2) / 2
#         J = np.abs(w1 - w2) / 2
#         kappa_p = sum(fwhms)
#     else:
#         # For single dip fallback: approximate coupled with small J and wp ≈ wr
#         wr = ws[w0_ind_list[0]]
#         wp = wr  # Or offset slightly if needed
#         J = fwhms[0] / 10  # Small coupling guess
#         kappa_p = fwhms[0]
    
#     # Phi sign: use the span across all dips
#     overall_fwhm = np.max(ws) - np.min(ws) if nr_dips == 1 else (w2 - w1) + max(fwhms)
#     m = np.abs(ws - wr) < overall_fwhm / 2
#     if np.any(m):
#         phi_sign = np.sign(s21s_abs_tilted_shifted[m][-1] - s21s_abs_tilted_shifted[m][0])
#     else:
#         phi_sign = 1
#     phi = phi_sign * phi_abs
    
#     return {
#         'A': a,
#         'k': b,
#         'phi': phi,
#         'kappa_p': kappa_p,
#         'omega_p': wp,
#         'omega_r': wr,
#         'J': J
#     }

# def coupled_hanger(w, A, k, phi, kappa_p, omega_p, omega_r, J, **kw):
#     """
#     Coupled mode model from eq (2.6). Returns |S_out,in(w)|.
#     omega_0 is fixed to w.mean().
#     """
#     omega_0 = np.mean(w)
#     Delta_p = w - omega_p
#     Delta_r = w - omega_r
    
#     numer = kappa_p * (-2j * Delta_p)
#     denom = 4 * J**2 + (kappa_p - 2j * Delta_p) * (-2j * Delta_r)
#     frac = numer / denom
    
#     s = np.cos(phi) - np.exp(1j * phi) * frac
#     s = (A + k * (w - omega_0)) * s
#     return np.abs(s)

# def fit_resonance_coupled(
#     ws: Optional[np.ndarray] = None, 
#     s21s: Optional[np.ndarray] = None, 
#     manual_graph_guess: Optional[dict] = None
# ) -> Tuple[dict, np.ndarray]:
#     """
#     Fit function for coupled mode, parallel to fit_resonance.
#     """
#     graph_guess = graphical_guess_coupled(ws / w_scale, s21s)  # type: ignore
#     if manual_graph_guess is not None:
#         graph_guess.update(**manual_graph_guess)
    
#     # Fit model
#     model = lmfit.Model(coupled_hanger)
#     params = model.make_params(
#         A=graph_guess['A'],
#         k=graph_guess['k'],
#         phi=graph_guess['phi'],
#         kappa_p=graph_guess['kappa_p'],
#         omega_p=graph_guess['omega_p'],
#         omega_r=graph_guess['omega_r'],
#         J=graph_guess['J']
#     )
    
#     # Set bounds
#     for param_name in ['kappa_p', 'J', 'omega_p', 'omega_r']:
#         params[param_name].min = 0
    
#     # Mask to region around features (e.g., 50 * kappa_p)
#     overall_span = 50 * graph_guess['kappa_p']
#     center = (graph_guess['omega_p'] + graph_guess['omega_r']) / 2
#     m = np.abs(ws / w_scale - center) < overall_span
    
#     fit_results_dict = {}
    
#     try:
#         result = model.fit(w=ws[m] / w_scale, data=np.abs(s21s[m]), params=params)
#         fit_results_dict.update(**result.best_values)
#     except (ValueError, TypeError) as e:
#         print(f'Fit failed: {e}')
#         print('Using graphical guess instead.')
#         fit_results_dict.update(**graph_guess)
    
#     # Compute fitted trace
#     s21_fit_trace = coupled_hanger(ws / w_scale, **fit_results_dict)
    
#     # Scale back frequencies and rates
#     for key in ['kappa_p', 'omega_p', 'omega_r', 'J']:
#         if key in fit_results_dict:
#             fit_results_dict[key] *= w_scale
    
#     return fit_results_dict, s21_fit_trace

def print_large_text(text):
    letters = {
        'A': ["  #  ",
              " # # ",
              "#####",
              "#   #",
              "#   #"],
        'B': ["#### ",
              "#   #",
              "#### ",
              "#   #",
              "#### "],
        'C': ["#### ",
              "#    ",
              "#    ",
              "#    ",
              "#### "],
        'D': ["#### ",
              "#   #",
              "#   #",
              "#   #",
              "#### "],
        'E': ["#####",
              "#    ",
              "#####",
              "#    ",
              "#####"],
        'F': ["#####",
              "#    ",
              "#####",
              "#    ",
              "#    "],
        'G': ["#### ",
              "#    ",
              "# ###",
              "#   #",
              "#### "],
        'H': ["#   #",
              "#   #",
              "#####",
              "#   #",
              "#   #"],
        'I': ["#####",
              "  #  ",
              "  #  ",
              "  #  ",
              "#####"],
        'J': ["#####",
              "   # ",
              "   # ",
              "#  # ",
              " ##  "],
        'K': ["#  # ",
              "# #  ",
              "##   ",
              "# #  ",
              "#  # "],
        'L': ["#    ",
              "#    ",
              "#    ",
              "#    ",
              "#####"],
        'M': ["#   #",
              "## ##",
              "# # #",
              "#   #",
              "#   #"],
        'N': ["#   #",
              "##  #",
              "# # #",
              "#  ##",
              "#   #"],
        'O': [" ### ",
              "#   #",
              "#   #",
              "#   #",
              " ### "],
        'P': ["#### ",
              "#   #",
              "#### ",
              "#    ",
              "#    "],
        'Q': [
            " ### ",
            "#   #",
            "#   #",
            "#  ##",
            " ## #"],
        'R': ["#### ",
              "#   #",
              "#### ",
              "#  # ",
              "#   #"],
        'S': [" ### ",
              "#    ",
              " ### ",
              "    #",
              "#### "],
        'T': ["#####",
              "  #  ",
              "  #  ",
              "  #  ",
              "  #  "],
        'U': ["#   #",
              "#   #",
              "#   #",
              "#   #",
              " ### "],
        'V': ["#   #",
              "#   #",
              "#   #",
              " # # ",
              "  #  "],
        'W': ["#   #",
              "#   #",
              "# # #",
              "## ##",
              "#   #"],
        'X': ["#   #",
              " # # ",
              "  #  ",
              " # # ",
              "#   #"],
        'Y': ["#   #",
              " # # ",
              "  #  ",
              "  #  ",
              "  #  "],
        'Z': ["#####",
              "   # ",
              "  #  ",
              " #   ",
              "#####"],
        ' ': ["     ",
              "     ",
              "     ",
              "     ",
              "     "],
      '<': [
              "   ##",
              "  ## ",
              " ##  ",
              "  ## ",
              "   ##"],        
      '3': [
              "###  ",
              "   # ",
              " ##  ",
              "   # ",
              "###  "
              ]
    }

    for row in range(5):
        line = ''
        for char in text.upper():
            if char in letters:
                line += letters[char][row] + '  '
            else:
                line += '     '  
        print(line)