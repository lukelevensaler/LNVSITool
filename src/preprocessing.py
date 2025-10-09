# Basic Imports
import logging
import warnings

# Data Science Imports
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.special import wofz
from skopt import gp_minimize
from skopt.space import Integer
from fastdtw import fastdtw # type: ignore
# from lmfit.models import VoigtModel done at deconvolution runtime

# Core Imports
from config import LOG_FILE
from utils import ErrorManager # error pausing checks for every error-related QMessageBox instance in this file specifically

# PyQt6 GUI Imports
from PyQt6.QtWidgets import QMessageBox, QApplication

# Import QApplication only when needed to avoid circular import issues
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	pass
 
# Suppress a noisy but non-fatal warning coming from the 'uncertainties' package
# which some downstream libraries may use when an uncertainty's std_dev==0.
warnings.filterwarnings(
	"ignore",
	message=r"Using UFloat objects with std_dev==0 may give unexpected results\.",
	category=UserWarning,
)
 
logging.basicConfig(
	filename= LOG_FILE,
	level=logging.DEBUG,
	format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d'
)
class Preprocessor:
    def __init__(self, ui=None, rh=None):
        self.ui = ui
        self.rh = rh  # Reference to the main application handler for error recovery
        self.wavelengths: np.ndarray | None = None
    
    def set_ui_and_rh(self, ui, rh):
        """Set UI and RH references after initialization to avoid circular dependencies."""
        self.ui = ui
        self.rh = rh
        
    def bayesian_savgol_optimize_win_and_poly(self, data_dict, wavelengths):
        """
        Dynamically determine optimal Savitzky-Golay window and polynomial order
        across the entire dataset using Bayesian optimization.
        
        Args:
            data_dict: Dictionary mapping sample names to their absorbance arrays
            wavelengths: Array of wavelength values
            
        Returns:
            tuple: (optimal_window, optimal_polyorder)
        """
        try:
            logging.info("Starting Bayesian optimization for SG parameters across entire dataset.")
            QApplication.processEvents()
            
            # Collect all y-arrays from the dataset
            all_y_arrays = []
            min_length = float('inf')
            
            for name, y_data in data_dict.items():
                y = np.asarray(y_data)
                if len(y) > 0:
                    all_y_arrays.append(y)
                    min_length = min(min_length, len(y))
            
            if not all_y_arrays or min_length == 0:
                logging.warning("No valid data arrays found for optimization, using defaults.")
                return 11, 2
            
            # Define objective function that evaluates across all samples
            def sg_error_global(params):
                win = int(params[0])
                if win % 2 == 0:
                    win += 1
                poly = int(params[1])
                
                # Validate parameters
                if win <= poly or win < 5 or poly < 2:
                    return np.inf
                
                total_error = 0.0
                valid_samples = 0
                
                for y in all_y_arrays:
                    try:
                        # Ensure window doesn't exceed signal length
                        actual_win = min(win, len(y) - 1 if len(y) % 2 == 1 else len(y) - 2)
                        if actual_win < 5:
                            continue
                        
                        # Ensure polynomial order is valid for this window
                        actual_poly = min(poly, actual_win - 1)
                        if actual_poly < 2:
                            continue
                        
                        y_sg = savgol_filter(y, window_length=actual_win, polyorder=actual_poly)
                        
                        # Use MSE as optimization criterion
                        mse = np.mean((y - y_sg) ** 2)
                        total_error += mse
                        valid_samples += 1
                        
                    except Exception:
                        # Skip this sample if filtering fails
                        continue
                
                if valid_samples == 0:
                    return np.inf
                
                # Return average MSE across all valid samples
                return total_error / valid_samples
            
            # Set optimization bounds based on minimum signal length and dataset characteristics
            max_win_cap = max(5, min(101, int(min_length * 0.2)))  # 20% of shortest signal
            if max_win_cap % 2 == 0:
                max_win_cap -= 1
            
            # Conservative bounds to prevent oversmoothing
            win_lower = 5
            win_upper = max(5, max_win_cap)
            poly_lower = 2
            poly_upper = min(5, max(2, win_upper - 1))
            
            search_space = [
                Integer(win_lower, win_upper),
                Integer(poly_lower, poly_upper)
            ]
            
            logging.info(f"Optimizing SG parameters with bounds: window [{win_lower}, {win_upper}], poly [{poly_lower}, {poly_upper}]")
            
            # Run Bayesian optimization with more calls for better convergence
            result = gp_minimize(
                sg_error_global, 
                search_space, 
                n_calls=25,  # Increased for better optimization across dataset
                random_state=0,
                acq_func='EI'  # Expected Improvement acquisition function
            )
            
            if result is not None and hasattr(result, 'x') and len(result.x) >= 2:
                win_opt = int(result.x[0])
                if win_opt % 2 == 0:
                    win_opt += 1
                poly_opt = int(result.x[1])
                poly_opt = max(2, min(poly_opt, win_opt - 1))
                
                logging.info(f"Bayesian optimization completed. Optimal parameters: window={win_opt}, poly={poly_opt}")
                return win_opt, poly_opt
            else:
                logging.warning("Bayesian optimization failed, using conservative defaults.")
                return 11, 2
                
        except Exception as e:
            logging.error(f"Error in bayesian_savgol_optimize_win_and_poly: {e}")
            return 11, 2  # Return safe defaults on any error

    def smooth_curves(self, x, y, win_opt=None, poly_opt=None):
        try:
            # Simplified smoothing: use Savitzky-Golay directly on the original sampled data
            if self.ui is not None:
                self.ui.show_countdown_gif(self.ui.progress_bar_row, self.ui.progress_container, trigger_type='progress', duration_ms=3000)
            logging.info("Starting smooth_curves method (Savitzky-Golay only).")
            QApplication.processEvents()
            
            # Work on numpy arrays
            x = np.asarray(x)
            y = np.asarray(y)
            
            # Use provided optimal parameters if available, otherwise use individual optimization
            if win_opt is None or poly_opt is None:
                logging.info("No pre-computed SG parameters provided, running individual optimization.")
                
                # Define objective for SG optimization: minimize MSE between y and filtered y
                def sg_error(params):
                    win = int(params[0])
                    if win % 2 == 0:
                        win += 1
                    # Prevent oversmoothing: cap window to 0.2 * len(y) or len(y)-1 whichever smaller
                    max_cap = max(5, int(min(101, max(5, int(len(y) * 0.2)))))
                    if len(y) - 1 <= max_cap:
                        win = max(5, min(win, len(y)-1 if len(y)%2==1 else len(y)-2))
                    else:
                        win = max(5, min(win, max_cap))
                    poly = int(params[1])
                    poly = max(2, min(poly, win-1))
                    try:
                        y_sg = savgol_filter(y, window_length=win, polyorder=poly)
                        return float(np.mean((y - y_sg) ** 2))
                    except Exception:
                        return np.inf
                
                # Set bounds based on signal length, but cap window to 20% of length to avoid oversmoothing
                win_upper = max(5, min(101, max(5, int(len(y) * 0.2))))
                if win_upper % 2 == 0:
                    win_upper -= 1
                search_space = [Integer(5, max(5, win_upper)), Integer(2, min(5, max(2, win_upper-1)))]
                
                logging.info("Starting individual Bayesian optimization for Savitzky-Golay parameters.")
                result = gp_minimize(sg_error, search_space, n_calls=18, random_state=0)
                
                if result is not None and hasattr(result, 'x'):
                    win_opt = int(result.x[0])
                    if win_opt % 2 == 0:
                        win_opt += 1
                    poly_opt = int(result.x[1])
                    poly_opt = max(2, min(poly_opt, win_opt-1))
                else:
                    win_opt = 11
                    poly_opt = 2
            else:
                logging.info(f"Using pre-computed optimal SG parameters: window={win_opt}, poly={poly_opt}")
            
            # Bias smoothing slightly above the optimizer suggestion but keep smoothing weaker than deconvolution
            try:
                # Smaller bias so the SG baseline remains weaker than the deconvolution operation
                bias_factor = 1.2
                win_biased = int(max(5, min(len(y)-1, int(win_opt * bias_factor))))
                if win_biased % 2 == 0:
                    win_biased += 1
                # don't exceed reasonable cap
                if win_biased >= len(y):
                    win_biased = len(y) - 1 if (len(y) - 1) % 2 == 1 else len(y) - 2
                win_opt = max(5, win_biased)
            except Exception:
                # fallback to original
                pass
            if self.ui is not None and hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
                self.ui.progress_bar.setValue(30)
            if self.ui is not None and hasattr(self.ui, 'set_splash_text'):
                self.ui.set_splash_text()
            QApplication.processEvents()
            # Ensure window length is valid for savgol
            if win_opt >= len(y):
                win_opt = len(y) - 1 if (len(y) - 1) % 2 == 1 else len(y) - 2
            # Use a conservative polynomial order to avoid fitting noise
            polyorder = min(poly_opt, 3)
            y_final = savgol_filter(y, window_length=max(5, win_opt), polyorder=polyorder)
            if self.ui is not None:
                self.ui.remove_countdown_gif_and_timer()
            # Return original x, smoothed y, and sg parameters for downstream residual-aware fitting
            return x, y_final, int(win_opt), int(poly_opt)
        except Exception as e:
            if self.ui is not None:
                self.ui.remove_countdown_gif_and_timer()
            logging.error(f"Error in smooth_curve: {e}")
            if not ErrorManager.errors_suppressed and self.ui is not None:   
                QMessageBox.critical(self.ui, "Error Smoothing Curve", f"An error occurred while smoothing the spectrum.\n\nError: {e}")
            if self.ui is not None:
                self.ui.restore_cursor()
            if self.rh is not None:
                self.rh.return_home_from_error()
            # Return best-effort values and default SG params on failure
            return x, y, 11, 2

    def align_curves_dtw(self, ref, query):
        
        try:
            if self.ui is not None:
                self.ui.show_countdown_gif(self.ui.progress_bar_row, self.ui.progress_container, trigger_type='progress', duration_ms=3000)
            logging.info("Starting align_curves_dtw method.")
            QApplication.processEvents()
            distance, path = fastdtw(ref, query)
            logging.info(f"DTW alignment complete. Distance: {distance}")
            aligned = np.interp(np.arange(len(ref)), [p[1] for p in path], [query[p[1]] for p in path])
            
            if self.ui is not None and hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
                self.ui.progress_bar.setValue(60)
            
            if self.ui is not None and hasattr(self.ui, 'set_splash_text'):
                self.ui.set_splash_text()
            QApplication.processEvents()
            logging.info("Curves aligned using DTW.")
            if self.ui is not None:
                self.ui.remove_countdown_gif_and_timer()
            return aligned
        
        except Exception as e:
            if self.ui is not None:
                self.ui.remove_countdown_gif_and_timer()
            logging.error(f"Error in align_curves_dtw: {e}")
            if not ErrorManager.errors_suppressed and self.ui is not None:  
                QMessageBox.critical(self.ui, "Error Aligning Curves", f"An error occurred while aligning curves using DTW.\n\nError: {e}")
            if self.ui is not None:
                self.ui.restore_cursor()
            if self.rh is not None:
                self.rh.return_home_from_error()
            return query

    def deconvolve_voigt(self, x, y, center_nm, window=30, baseline=None, sg_win=None, sg_poly=None):
        """
        Deconvolve (fit) peaks in a window centered at center_nm +/- window.
        Returns an array of same length as y with the fitted values on the window
        and np.nan elsewhere so caller can reconstruct the full spectrum.
        """
        try:
            if self.ui is not None:
                self.ui.show_countdown_gif(self.ui.progress_bar_row, self.ui.progress_container, trigger_type='progress', duration_ms=3000)
            logging.info(f"Starting deconvolve_voigt around {center_nm} +/- {window} nm.")
            QApplication.processEvents()
            x = np.asarray(x)
            y = np.asarray(y)
            mask = (x >= (center_nm - window)) & (x <= (center_nm + window))
            if not np.any(mask):
                logging.warning(f"No data in window for center {center_nm} nm; returning original segment.")
                if self.ui is not None:
                    self.ui.remove_countdown_gif_and_timer()
                out = np.full_like(y, np.nan, dtype=float)
                return out
            x_seg = x[mask]
            y_seg = y[mask]
            # If baseline (smoothed) provided, fit to the baseline to preserve smoothing
            if baseline is not None:
                baseline_seg = np.asarray(baseline[mask], dtype=float)
                # Fit target is the smoothed baseline so fits don't undo smoothing
                y_fit_target = baseline_seg.copy()
            else:
                # No baseline provided, fit directly to the observed segment
                y_fit_target = np.asarray(y_seg, dtype=float)

            # Find local maxima in the fit target within the window (used for initial center guesses)
            try:
                # Increase sensitivity: lower height threshold to detect smaller peaks within the window
                peaks, _ = find_peaks(y_fit_target, height=np.max(y_fit_target) * 0.02, distance=max(1, int(float(window)*0.05)))
            except Exception:
                peaks = np.array([], dtype=int)

            # Determine which peak is closest to center_nm in x-space
            peak_idx = None
            if peaks.size > 0:
                # convert indices to wavelengths
                peak_waves = x_seg[peaks]
                closest = np.argmin(np.abs(peak_waves - center_nm))
                peak_idx = peaks[closest]

            # For 779 doublet, we may have two peaks; we'll try to detect up to 2 peaks
            if center_nm == 779:
                n_peaks = 2
            else:
                n_peaks = 1
            # For narrow windows, attempt single or double peak depending on center (779 doublet)
            if center_nm == 779:
                n_peaks = 2
            else:
                n_peaks = 1

            amps = np.maximum(y_fit_target, 0)

            # Voigt profile helper
            def voigt_profile(xv, amp, cen, sigma, gamma):
                # ensure float operations
                xv = np.array(xv, dtype=float)
                z = ((xv - cen) + 1j * gamma) / (sigma * np.sqrt(2.0))
                v = amp * np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
                return v

            def multi_voigt(xa, *params):
                res = np.zeros_like(xa, dtype=float)
                for i in range(n_peaks):
                    amp = params[i*4]
                    cen = params[i*4+1]
                    sig = params[i*4+2]
                    gam = params[i*4+3]
                    res += voigt_profile(xa, amp, cen, sig, gam)
                return res

            p0 = []
            bounds_low = []
            bounds_high = []
            centers = []
            if n_peaks == 1:
                # If a peak was located near center_nm, use its wavelength, else use center_nm
                if peak_idx is not None:
                    centers = [float(x_seg[peak_idx])]
                else:
                    centers = [center_nm]
            else:
                # For doublet, attempt to use two highest peaks if available, else fallback to offsets
                if peaks.size >= 2:
                    # pick two peaks closest to center
                    sorted_peaks = peaks[np.argsort(np.abs(x_seg[peaks] - center_nm))][:2]
                    centers = [float(x_seg[p]) for p in sorted_peaks]
                else:
                    centers = [center_nm - 6, center_nm + 6]

            for i, c in enumerate(centers):
                # amplitude guess: use the smoothed fit target (baseline or local smoothed data)
                # derive local mask spans from the caller-provided window (in nm)
                # local_amp_half: small neighborhood for amplitude estimation
                # local_fit_half: slightly larger neighborhood for using lmfit.guess/local fitting
                local_amp_half = max(1.0, float(window) * 0.15)
                local_fit_half = max(2.0, float(window) * 0.25)
                local_mask_amp = (x_seg >= (c - local_amp_half)) & (x_seg <= (c + local_amp_half))
                if np.any(local_mask_amp):
                    local_max = float(np.max(y_fit_target[local_mask_amp]))
                else:
                    local_max = float(np.max(y_fit_target)) if y_fit_target.size>0 else float(np.max(y_seg))
                amp_guess = max(1e-6, 0.05 * local_max, local_max)
                # p0: amp, cen, sigma, gamma
                # initial p0: amplitude, center, sigma, gamma
                p0 += [amp_guess, float(c), max(0.8, (x_seg.max()-x_seg.min())/20.0), max(0.8, (x_seg.max()-x_seg.min())/20.0)]
                
                # We favor broader peaks: set safer minima for sigma/gamma so fits are less sharp
                """
                THIS IS CRITICAL !!!
                - AVOIDS SPIKE-LIKE ARTIFACTS ON SIGNIFICANT FEATURES 
                - AVOIDS ERRONEOUS AMPLIFICATION OF INSIGNIFICANT FEATURES
                """
                bounds_low += [0, c - 10, 0.8, 0.8]
                bounds_high += [max(amp_guess * 20 + 1, np.max(y_seg) * 5 + 1), c + 10, (x_seg.max()-x_seg.min()), (x_seg.max()-x_seg.min())]

            # Fit to residual using lmfit VoigtModel(s) ONLY. On any failure return NaNs for this window.
            try:
                # ensure float arrays for lmfit
                x_fit = np.array(x_seg, dtype=float)
                y_fit = np.array(y_fit_target, dtype=float)
                if y_fit.size == 0:
                    raise ValueError("Empty fit region")

                # Build composite lmfit model with 1 or 2 Voigt components
                from lmfit.models import VoigtModel
                composite = None
                for i, c in enumerate(centers):
                    prefix = f'v{i}_'
                    model = VoigtModel(prefix=prefix)
                    if composite is None:
                        composite = model
                    else:
                        composite = composite + model

                if composite is None:
                    raise RuntimeError("No Voigt model constructed for fitting")

                # Create parameters and attempt robust guesses per-component using lmfit's guess helper
                params = composite.make_params()
                for i, c in enumerate(centers):
                    prefix = f'v{i}_'
                    # Prepare a local window around the component to get a better initial guess
                    local_mask = (x_fit >= (c - local_fit_half)) & (x_fit <= (c + local_fit_half))
                    if np.any(local_mask) and np.sum(local_mask) >= 3:
                        x_local = x_fit[local_mask]
                        y_local = y_fit[local_mask]
                    else:
                        x_local = x_fit
                        y_local = y_fit
                    # Try using VoigtModel.guess for initial parameters; fall back to conservative manual guesses
                    try:
                        g = VoigtModel(prefix=prefix).guess(y_local, x=x_local)
                        # copy any guessed values into our composite params
                        for pname, pval in g.items():
                            if pname in params:
                                params[pname].set(value=pval.value)
                    except Exception:
                        # Conservative manual initialization but bias towards broader (less sharp) peaks
                        local_max = float(np.max(y_local)) if np.any(y_local) else float(np.max(y_fit))
                        params[f'{prefix}amplitude'].set(value=max(1e-6, local_max), min=0, max=max(1e-6, local_max * 6 + 1e-6))
                        params[f'{prefix}center'].set(value=float(c), min=float(c - 20), max=float(c + 20))
                        # prefer broader components to avoid spike-like artifacts
                        init_sigma = max(0.8, (x_fit.max() - x_fit.min()) / 20.0)
                        params[f'{prefix}sigma'].set(value=init_sigma, min=0.8, max=float(x_fit.max() - x_fit.min()))
                        params[f'{prefix}gamma'].set(value=init_sigma, min=0.8, max=float(x_fit.max() - x_fit.min()))

                # Final safety: ensure bounds exist for each param and keep fits local to the window
                for i, c in enumerate(centers):
                    prefix = f'v{i}_'
                    if f'{prefix}amplitude' in params:
                        # limit amplitude upper bound to reduce spike risk
                        current = params[f'{prefix}amplitude']
                        if current.max is None:
                            current.set(min=0, max=max(1e-6, float(current.value) * 6 + 1e-6))
                        else:
                            current.set(min=current.min if current.min is not None else 0, max=max(float(current.max), float(current.value) * 3 + 1e-6))
                    if f'{prefix}center' in params:
                        # keep center within a reasonable fraction of the analysis window to avoid drifting into baseline
                        center_shift = max(3.0, float(window) * 0.25)
                        cmin = float(params[f'{prefix}center'].min if params[f'{prefix}center'].min is not None else c - center_shift)
                        cmax = float(params[f'{prefix}center'].max if params[f'{prefix}center'].max is not None else c + center_shift)
                        params[f'{prefix}center'].set(min=cmin, max=cmax)
                    if f'{prefix}sigma' in params:
                        # enforce broader minima for sigma/gamma to reduce sharp spikes
                        max_sigma = float(max(0.8, min((x_fit.max() - x_fit.min()) / 2.0, (sg_win if sg_win is not None else window) )))
                        params[f'{prefix}sigma'].set(min=0.8, max=max_sigma)
                    if f'{prefix}gamma' in params:
                        max_sigma = float(max(0.8, min((x_fit.max() - x_fit.min()) / 2.0, (sg_win if sg_win is not None else window) )))
                        params[f'{prefix}gamma'].set(min=0.8, max=max_sigma)

                # Try multiple lmfit solvers in order until a reasonable fit is found. All attempts use lmfit only.
                fit_methods = ['leastsq', 'least_squares', 'nelder']
                result = None
                last_err = None
                for method in fit_methods:
                    try:
                        # Use lmfit defaults (no inverse-amplitude weighting) to avoid amplifying noise
                        res = composite.fit(y_fit, params, x=x_fit, method=method, max_nfev=10000, nan_policy='omit')
                        # Accept if lmfit reports success or a reasonable reduced chi-square
                        redchi_ok = hasattr(res, 'redchi') and (res.redchi is not None) and (res.redchi < 100)
                        if getattr(res, 'success', False) or redchi_ok:
                            result = res
                            break
                        last_err = None
                    except Exception as me:
                        last_err = me
                        logging.debug(f"lmfit method {method} raised: {me}")

                if result is None:
                    # If no method produced a usable result, raise to trigger NaN policy
                    raise RuntimeError(f"lmfit Voigt fit did not converge for center {center_nm}; last_err={last_err}")

                # Reconstruct fitted model from lmfit params but enforce sensible minima/limits
                # to avoid extremely narrow, tall spikes while preserving peak topology.
                local_max_global = float(np.max(y_fit)) if y_fit.size>0 else float(np.max(y_seg))
                # Enforce broader minima to reduce sharp spikes
                sigma_min_enforced = max(0.9, float(window) * 0.02)
                gamma_min_enforced = max(0.9, float(window) * 0.02)
                amp_mult_allowed = 4.0
                amp_limit = max(local_max_global * amp_mult_allowed, float(np.max(y_seg)) * 3.0 + 1.0)

                reconstructed_params = []
                for i in range(len(centers)):
                    prefix = f'v{i}_'
                    # Extract values safely from result.params
                    try:
                        amp_val = float(result.params[f'{prefix}amplitude'].value)
                    except Exception:
                        amp_val = float(result.params.get(f'{prefix}amplitude', {'value': 0}).get('value', 0))
                    try:
                        cen_val = float(result.params[f'{prefix}center'].value)
                    except Exception:
                        cen_val = centers[i]
                    try:
                        sig_val = float(result.params[f'{prefix}sigma'].value)
                    except Exception:
                        sig_val = (x_fit.max() - x_fit.min()) / 10.0
                    try:
                        gam_val = float(result.params[f'{prefix}gamma'].value)
                    except Exception:
                        gam_val = sig_val

                    # Enforce minima for widths to avoid spike-like fits
                    sig_val = max(sig_val, sigma_min_enforced)
                    gam_val = max(gam_val, gamma_min_enforced)

                    # Cap amplitude relative to local signal to prevent runaway spikes when widths are small
                    if amp_val < 0:
                        amp_val = 0.0
                    amp_val = min(amp_val, amp_limit)

                    reconstructed_params += [amp_val, cen_val, sig_val, gam_val]

                # Build fitted residual via multi_voigt (preserves voigt topology)
                try:
                    fitted_resid = multi_voigt(x_fit, *reconstructed_params).astype(float)
                except Exception:
                    # Fall back to lmfit's best_fit if reconstruction fails
                    fitted_resid = result.best_fit.astype(float)

                # Optional small clamp to avoid negative trough artifacts; preserve topology by allowing small negatives
                min_floor = -0.02 * local_max_global
                fitted_resid = np.clip(fitted_resid, min_floor, None)

                # Blend reconstructed fitted component into baseline; blend_alpha controls prominence
                blend_alpha = 0.8
                if baseline is not None:
                    fitted = baseline_seg.astype(float) + blend_alpha * fitted_resid
                else:
                    fitted = blend_alpha * fitted_resid
                logging.info(f"lmfit Voigt residual fit finished for center {center_nm} nm. Method used: {result.method if hasattr(result,'method') else 'unknown'}; success={getattr(result,'success',None)}")
            except Exception as fit_err:
                logging.error(f"lmfit Voigt fit failed for center {center_nm}: {fit_err}")
                # Preserve smoothing on fit failures: return the supplied smoothed baseline segment
                if baseline is not None:
                    # baseline_seg was computed earlier from the provided baseline
                    fitted = baseline_seg.astype(float).copy()
                else:
                    # No baseline provided: compute a small local Savitzky-Golay smoothing and return that
                    try:
                        win = int(sg_win) if (sg_win is not None and int(sg_win) > 3) else min(11, max(5, len(y_seg)//10))
                        if win % 2 == 0:
                            win += 1
                        poly = int(sg_poly) if (sg_poly is not None and int(sg_poly) >= 2) else 2
                        # Ensure window < len(y_seg)
                        if win >= len(y_seg):
                            win = len(y_seg) - 1 if (len(y_seg) - 1) % 2 == 1 else len(y_seg) - 2
                        fitted = savgol_filter(y_seg.astype(float), window_length=max(5, win), polyorder=min(poly, max(2, win-1)))
                    except Exception:
                        fitted = y_seg.astype(float).copy()

            out = np.full_like(y, np.nan, dtype=float)
            out[mask] = fitted
            if self.ui is not None and hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
                self.ui.progress_bar.setValue(50)
            if self.ui is not None and hasattr(self.ui, 'set_splash_text'):
                self.ui.set_splash_text()
            QApplication.processEvents()
            if self.ui is not None:
                self.ui.remove_countdown_gif_and_timer()
            return out
        except Exception as e:
            if self.ui is not None:
                self.ui.remove_countdown_gif_and_timer()
            logging.error(f"Error in deconvolve_voigt (windowed): {e}")
            if not ErrorManager.errors_suppressed and self.ui is not None:
                QMessageBox.warning(self.ui, "Voigt/Gauss Fit Warning", f"Fitting failed around {center_nm} nm. Using original data in that window.\n\nError: {e}")
            return np.full_like(y, np.nan, dtype=float)