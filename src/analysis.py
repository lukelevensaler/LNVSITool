# Basic Imports
import logging
import time
import json
import datetime
            
# Data Science Imports
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from scipy.fft import fft, ifft
from scipy.stats import pearsonr
from scipy.special import wofz
from skopt import gp_minimize
from skopt.space import Integer
from fastdtw import fastdtw
import decimal
import lmfit

# Machine Learning Imports
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# PyQt6 GUI Imports
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, 
    QMessageBox
)
from PyQt6.QtCore import QTimer, QElapsedTimer

# Core Imports
from utils import FDRUtils
from loop_manger import ReturnHome
from config import (
    qfiledialog__pinned_locations,
    LOG_FILE as global_logging_file,
    AUTO_SAVE_FILE as auto_save_file
)

logging.basicConfig(
    filename= global_logging_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d'
)

# Remove any import of main or UI to prevent circular import errors
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass  # from main import UI (for type hints only, not runtime)


def safe_number_for_json(val):

    """
    Convert a number to a JSON-safe representation 
    (string if too large/small)
    (prevents 64-bit int lim overflows).
    
    """

    if isinstance(val, (np.floating, float)):
        if np.isfinite(val):
            if abs(val) > 1e308 or (abs(val) < 1e-308 and val != 0):
                return repr(val)
            else:
                return float(val)
        else:
            return str(val)
    elif isinstance(val, (np.integer, int)):
        return int(val)
    elif isinstance(val, decimal.Decimal):
        return str(val)
    else:
        return val


class AnalysisEngine:

    def __init__(self, ui):
        self.filename = None
        self.ui = ui  # We use the passed UI instance directly
        self.wavelengths = None
        self.absorbance = None
        self.positive_control = None
        self.rh = ReturnHome(ui)
        logging.info("AnalysisEngine initialized.")

    def upload_csv(self):
        try:
            logging.info("Starting upload_csv method.")
            QApplication.processEvents()
            options = QFileDialog.Option.ReadOnly | QFileDialog.Option.DontUseNativeDialog
            file_dialog = QFileDialog(self.ui, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)")
            file_dialog.setOptions(options)
            sidebar_locations = qfiledialog__pinned_locations()
            
            if sidebar_locations:
                file_dialog.setSidebarUrls(sidebar_locations)
            
            if file_dialog.exec():
                file_name = file_dialog.selectedFiles()[0]
            else:
                file_name = ''
            logging.info("File dialog opened for CSV selection.")
            
            if not file_name:
                
                # User cancelled file dialog, re-enable upload button and exit
                if self.ui.upload_button is not None:
                    try:
                        self.ui.upload_button.setEnabled(True)
                    except Exception:
                        pass
                self.ui.restore_cursor()  # Restore cursor if user cancels
                return
            self.filename = file_name
            logging.info(f"CSV file selected: {self.filename}")
            
            # Show countdown gif/timer for 3 seconds after file selection
            self.ui.show_countdown_gif(self.ui.welcome_layout, self.ui.welcome_container, trigger_widget=self.ui.upload_button, trigger_type='progress', duration_ms=3000)
            
            # Launch analysis UI immediately
            self.ui.launch_data_analysis_mini_screen()
            logging.info("Data analysis mini screen launched.")
            self.ui.progress_bar = self.ui.analysis_progress_bar
            self._upload_timer = QElapsedTimer()
            self._upload_timer.start()
            logging.info("Upload timer started for artificial latency.")
            
            def finish_upload():
                QApplication.processEvents()
                if self._upload_timer.elapsed() >= 3000:
                    logging.info("Upload latency complete, proceeding to read_csv.")
                    self.read_csv()
                else:
                    QTimer.singleShot(100, finish_upload)
            QTimer.singleShot(100, finish_upload)
            logging.info("CSV file upload completed, waiting for latency to finish.")
            
            if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
                self.ui.progress_bar.setValue(0)
            
            if hasattr(self.ui, 'set_splash_text'):
                self.ui.set_splash_text()
            QApplication.processEvents()

        except Exception as e:
            logging.error(f"Error uploading CSV file: {e}")
            QMessageBox.critical(self.ui, "Error", f"Error uploading CSV file: {e}")
            self.ui.restore_cursor()
            raise RuntimeError(f"Error uploading CSV file: {e}")

    def read_csv(self):
        
        try:
            logging.info("Starting read_csv method.")
            QApplication.processEvents()
            df = pd.read_csv(self.filename)
            logging.info("CSV file loaded into DataFrame.")

            # Normalize column names for matching, filter out 'Unnamed:' and empty columns
            columns_clean = [str(col).strip().lower() for col in df.columns]
            col_map = {str(col).strip().lower(): col for col in df.columns}
            # Remove columns with names starting with 'unnamed:' or empty
            filtered_cols = [col for col in df.columns if str(col).strip() and not str(col).strip().lower().startswith('unnamed:')]
            df = df[filtered_cols]
            columns_clean = [str(col).strip().lower() for col in df.columns]
            col_map = {str(col).strip().lower(): col for col in df.columns}
            logging.info(f"Filtered columns: {list(df.columns)}")
            if 'nm' not in columns_clean:
                raise ValueError("CSV must contain an 'nm' column for wavelengths.")
            if 'positive control' not in columns_clean:
                raise ValueError("CSV must contain a 'positive control' column.")
            nm_col = col_map['nm']
            pos_ctrl_col = col_map['positive control']
            analyte_cols = [col_map[c] for c in columns_clean if c not in ('nm', 'positive control')]
            # Remove duplicates
            seen = set()
            unique_analyte_cols = []
            for col in analyte_cols:
                if col not in seen:
                    unique_analyte_cols.append(col)
                    seen.add(col)
            analyte_cols = unique_analyte_cols
            if len(analyte_cols) < 1:
                raise ValueError("CSV must contain at least one analyte column in addition to 'nm' and 'positive control'.")
            # Reorder DataFrame
            new_order = [nm_col, pos_ctrl_col] + analyte_cols
            df = df[new_order]
            logging.info(f"Columns reordered to: {new_order}")
            wavelengths = df.iloc[:, 0].values
            samples = [pos_ctrl_col] + analyte_cols
            valid_samples = []
            for name in samples:
                col = df[name]
                if not pd.api.types.is_numeric_dtype(col):
                    logging.warning(f"Skipping column '{name}': non-numeric data.")
                    continue
                if len(col) != len(wavelengths):
                    logging.warning(f"Skipping column '{name}': length {len(col)} does not match nm column {len(wavelengths)}.")
                    continue
                valid_samples.append(name)
            if len(valid_samples) < 2:
                raise ValueError("CSV must contain at least 'positive control' and one analyte column after 'nm'.")
            # CHANGED: Only store 1D absorbance arrays, not tuples
            data = {name: df[name].values for name in valid_samples}
            self.wavelengths = wavelengths
            self.absorbance = data
            self.positive_control = valid_samples[0]
            self.sample_names = valid_samples[1:]
            self.num_analytes = len(self.sample_names)
            logging.info(f"Positive control: {self.positive_control}. Analytes: {self.sample_names}")
            self.processed = self.preprocess_all_curves(data)
            if self.positive_control not in self.processed:
                raise ValueError(f"Positive control '{self.positive_control}' not found in processed data after preprocessing.")
            if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
                self.ui.progress_bar.setValue(10)
            if hasattr(self.ui, 'set_splash_text'):
                self.ui.set_splash_text()
            QApplication.processEvents()
            logging.info(f"CSV file read successfully: {self.filename} | {self.num_analytes} analytes found. | Beginning smoothing..")
            logging.info("Proceeding to analyze_all_samples after preprocessing.")
            self.ui._cached_results = self.analyze_all_samples()
            logging.info("analyze_all_samples completed. Results returned.")
        except Exception as e:
            logging.error(f"Error in read_csv: {e}")
            QMessageBox.critical(self.ui, "Error Reading CSV", f"An error occurred while reading your CSV file. Please check the format and try again.\n\nError: {e}")
            self.ui.restore_cursor()
            self.rh.return_home_from_error()
            return

    def preprocess_all_curves(self, data):
        try:
            logging.info("Starting preprocess_all_curves method.")
            QApplication.processEvents()
            processed = {}
            start_time = time.time()
            timeout = 60  # seconds
            if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
                self.ui.progress_bar.setValue(20)
            if hasattr(self.ui, 'set_splash_text'):
                self.ui.set_splash_text()
            QApplication.processEvents()
            for name, y in data.items():
                if time.time() - start_time > timeout:
                    logging.error("Preprocessing timed out.")
                    raise TimeoutError("Preprocessing timed out.")
                x = np.asarray(self.wavelengths)
                y = np.asarray(y)
                if x is None or len(x) != len(y):
                    logging.error(f"Length mismatch for '{name}': x({len(x) if x is not None else 'None'}), y({len(y)})")
                    raise ValueError(f"Length mismatch for '{name}': x({len(x) if x is not None else 'None'}), y({len(y)})")
                x_smooth, y_smooth = self.smooth_curve(x, y)
                processed[name] = (x_smooth, y_smooth)  # Store tuple (x, y)
            if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
                self.ui.progress_bar.setValue(35)
            if hasattr(self.ui, 'set_splash_text'):
                self.ui.set_splash_text()
            QApplication.processEvents()
            logging.info("All curves smoothed. Exiting preprocess_all_curves.")
            return processed
        except Exception as e:
            logging.error(f"Error in preprocess_all_curves: {e}")
            QMessageBox.critical(self.ui, "Error Preprocessing Data", f"An error occurred while preprocessing your data.\n\nError: {e}")
            self.ui.restore_cursor()
            self.rh.return_home_from_error()
            return {}

    def smooth_curve(self, x, y):
        try:
            self.ui.show_countdown_gif(self.ui.progress_bar_row, self.ui.progress_container, trigger_type='progress', duration_ms=3000)
            logging.info("Starting smooth_curve method.")
            QApplication.processEvents()
            def lowpass_filter(y, cutoff_ratio=0.1):
                logging.info("Applying lowpass filter.")
                N = len(y)
                yf = fft(y)
                cutoff = int(N * cutoff_ratio)
                yf[cutoff:-cutoff] = 0
                if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
                    self.ui.progress_bar.setValue(25)
                if hasattr(self.ui, 'set_splash_text'):
                    self.ui.set_splash_text()
                QApplication.processEvents()
                logging.info("Lowpass filter applied.")
                return np.real(ifft(yf))
            cs = CubicSpline(x, y)
            logging.info("CubicSpline interpolation complete.")
            x_new = np.linspace(x.min(), x.max(), 1000)
            y_smooth = cs(x_new)
            logging.info("Generated new x and smoothed y values.")
            y_lp = lowpass_filter(y_smooth)
            QApplication.processEvents()
            logging.info("Lowpass filtered y values ready.")
            def sg_error(params):
                logging.info("Starting Savitzky-Golay error calculation for perfect optimization of parameters.")
                win = int(params[0])
                if win % 2 == 0:
                    win += 1
                win = max(5, min(win, len(y_lp)-1 if len(y_lp)%2==1 else len(y_lp)-2))
                poly = int(params[1])
                poly = max(2, min(poly, win-1))
                try:
                    y_sg = savgol_filter(y_lp, window_length=win, polyorder=poly)
                    return np.mean((y_lp - y_sg)**2)
                except Exception:
                    return np.inf
            win_max = min(101, len(y_lp)-1)
            if win_max % 2 == 0:
                win_max -= 1
            search_space = [Integer(5, win_max), Integer(2, 5)]
            logging.info("Starting Bayesian optimization for Savitzky-Golay parameters.")
            opt_start = time.time()
            result = gp_minimize(sg_error, search_space, n_calls=20, random_state=0)
            if time.time() - opt_start > 30:
                logging.warning("Savitzky-Golay optimization is taking unusually long (possible infinite loop).")
                QMessageBox.warning(self.ui, "Warning", "Smoothing is taking unusually long. Please check your data or restart the app.")
            win_opt = int(result.x[0])
            if win_opt % 2 == 0:
                win_opt += 1
            poly_opt = int(result.x[1])
            poly_opt = max(2, min(poly_opt, win_opt-1))
            if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
                self.ui.progress_bar.setValue(30)
            if hasattr(self.ui, 'set_splash_text'):
                self.ui.set_splash_text()
            QApplication.processEvents()
            y_final = savgol_filter(y_lp, window_length=win_opt, polyorder=poly_opt)
            self.ui.remove_countdown_gif_and_timer()
            return x_new, y_final  # Return both x and y arrays
        except Exception as e:
            self.ui.remove_countdown_gif_and_timer()
            logging.error(f"Error in smooth_curve: {e}")
            QMessageBox.critical(self.ui, "Error Smoothing Curve", f"An error occurred while smoothing the spectrum.\n\nError: {e}")
            self.ui.restore_cursor()
            self.rh.return_home_from_error()
            return x, y

    def align_curves_dtw(self, ref, query):
        
        try:
            self.ui.show_countdown_gif(self.ui.progress_bar_row, self.ui.progress_container, trigger_type='progress', duration_ms=3000)
            logging.info("Starting align_curves_dtw method.")
            QApplication.processEvents()
            align_start = time.time()
            distance, path = fastdtw(ref, query)
            
            if time.time() - align_start > 30:
                logging.warning("DTW alignment is taking unusually long (possible infinite loop).")
                QMessageBox.warning(self.ui, "Warning", "Curve alignment is taking unusually long. Please check your data or restart the app.")
            logging.info(f"DTW alignment complete. Distance: {distance}")
            aligned = np.interp(np.arange(len(ref)), [p[1] for p in path], [query[p[1]] for p in path])
            
            if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
                self.ui.progress_bar.setValue(60)
            
            if hasattr(self.ui, 'set_splash_text'):
                self.ui.set_splash_text()
            QApplication.processEvents()
            logging.info("Curves aligned using DTW.")
            self.ui.remove_countdown_gif_and_timer()
            return aligned
        
        except Exception as e:
            self.ui.remove_countdown_gif_and_timer()
            logging.error(f"Error in align_curves_dtw: {e}")
            QMessageBox.critical(self.ui, "Error Aligning Curves", f"An error occurred while aligning curves using DTW.\n\nError: {e}")
            self.ui.restore_cursor()
            self.rh.return_home_from_error()
            return query

    def deconvolve_voigt(self, x, y):
        
        try:
            self.ui.show_countdown_gif(self.ui.progress_bar_row, self.ui.progress_container, trigger_type='progress', duration_ms=3000)
            logging.info("Starting deconvolve_voigt method.")
            QApplication.processEvents()

            # Voigt function for lmfit
            def voigt_profile(x, amp, cen, sigma, gamma):
                try:
                    z = ((x - cen) + 1j * gamma) / (sigma * np.sqrt(2))
                    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
                except Exception as e:
                    logging.warning(f"Exception in Voigt function: {e}")
                    return np.full_like(x, np.nan)

            # lmfit implementation
            try:
                model = lmfit.Model(voigt_profile)
                amp_guess = float(np.max(y))
                cen_guess = float(x[np.argmax(y)])
                sigma_guess = float(np.std(x) / 5)
                gamma_guess = float(np.std(x) / 5)
                params = model.make_params(
                    amp=amp_guess,
                    cen=cen_guess,
                    sigma=sigma_guess,
                    gamma=gamma_guess
                )
                # Set reasonable bounds
                params['amp'].set(min=0)
                params['sigma'].set(min=1e-6, max=(x.max()-x.min()))
                params['gamma'].set(min=1e-6, max=(x.max()-x.min()))
                params['cen'].set(min=x.min(), max=x.max())
                result = model.fit(y, params, x=x)
                if not result.success or np.any(np.isnan(result.best_fit)):
                    raise RuntimeError(f"lmfit Voigt fit failed: {result.message}")
                fit = result.best_fit
                fit_success = True
                logging.info(f"lmfit Voigt fit succeeded. Fit report: {result.fit_report()}")
            except ImportError as e:
                logging.error("lmfit is not installed. Please install lmfit for robust Voigt fitting.")
                QMessageBox.warning(self.ui, "lmfit Not Installed", "The lmfit package is required for robust Voigt fitting. Falling back to scipy curve_fit.")
                fit_success = False
                fit = None
            except Exception as fit_err:
                logging.warning(f"lmfit Voigt fit failed with error: {fit_err}")
                fit_success = False
                fit = None

            if not fit_success:
                # Fallback: fit a simple Gaussian using scipy
                from scipy.optimize import curve_fit
                def gauss(x, amp, cen, sigma):
                    return amp * np.exp(-0.5 * ((x - cen) / sigma) ** 2)
                try:
                    popt, _ = curve_fit(gauss, x, y, p0=[np.max(y), x[np.argmax(y)], np.std(x)/5], maxfev=3000)
                    fit = gauss(x, *popt)
                    logging.info("Fallback Gaussian fit succeeded.")
                except Exception as gauss_err:
                    logging.warning(f"Fallback Gaussian fit also failed: {gauss_err}")
                    fit = y  # Use original data as fallback

            if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
                self.ui.progress_bar.setValue(50)
            if hasattr(self.ui, 'set_splash_text'):
                self.ui.set_splash_text()
            QApplication.processEvents()
            logging.info("Voigt (or fallback) fit complete.")
            self.ui.remove_countdown_gif_and_timer()
            return fit
        except Exception as e:
            self.ui.remove_countdown_gif_and_timer()
            logging.error(f"Error in deconvolve_voigt: {e}")
            if hasattr(self.ui, 'progress_label'):
                self.ui.progress_label.setText("Voigt fit failed, using original data...")
                QApplication.processEvents()
            QMessageBox.warning(self.ui, "Voigt Fit Warning", f"Voigt profile fitting failed. The original data will be used for this region.\n\nError: {e}")
            self.ui.close_all_message_boxes()
            self.ui.restore_cursor()
            self.rh.return_home_from_error()
            return y

    def compute_similarity_metrics(self, y_ctrl, y_sample):
        
        try:
            self.ui.show_countdown_gif(self.ui.progress_bar_row, self.ui.progress_container, trigger_type='progress', duration_ms=3000)
            logging.info("Starting compute_similarity_metrics method.")
            QApplication.processEvents()
            sim_start = time.time()
            scaler = StandardScaler()
            X = np.vstack([y_ctrl, y_sample])
            X_scaled = scaler.fit_transform(X)
            logging.info("Data standardized for similarity metrics.")
            pca = PCA(n_components=1)
            pca.fit(X_scaled)
            pca_score = np.abs(pca.components_[0][0] - pca.components_[0][1])
            
            if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
                self.ui.progress_bar.setValue(65)
            
            if hasattr(self.ui, 'set_splash_text'):
                self.ui.set_splash_text()
            QApplication.processEvents()
            logging.info(f"PCA score computed: {pca_score}")
            
            pls = PLSRegression(n_components=2)
            
            try:
                pls.fit(y_ctrl.reshape(-1, 1), y_sample)
                pls_score = pls.score(y_ctrl.reshape(-1, 1), y_sample)
                logging.info(f"PLS score computed: {pls_score}")
            except Exception:
                pls_score = 0
                logging.info("PLS score computation failed, set to 0.")
            
            cos_sim = cosine_similarity([y_ctrl], [y_sample])[0, 0]
            pearson_corr, _ = pearsonr(y_ctrl, y_sample)
            euclid_dist = np.linalg.norm(y_ctrl - y_sample)
            
            if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
                self.ui.progress_bar.setValue(70)
            
            if hasattr(self.ui, 'set_splash_text'):
                self.ui.set_splash_text()
            QApplication.processEvents()
            logging.info(f"Cosine similarity: {cos_sim}, Pearson: {pearson_corr}, Euclidean: {euclid_dist}")
            
            auc_diff = np.abs(np.trapezoid(y_ctrl) - np.trapezoid(y_sample)) / np.trapezoid(y_ctrl)
            sim_metrics = np.array([
                cos_sim,
                (pearson_corr + 1) / 2,
                1 / (1 + euclid_dist),
                1 - auc_diff,
                1 - pca_score,
                pls_score if pls_score > 0 else 0
            ])
            sim_score = np.clip(np.mean(sim_metrics), 0, 1) * 100
            
            if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
                self.ui.progress_bar.setValue(75)
            
            if hasattr(self.ui, 'set_splash_text'):
                self.ui.set_splash_text()
            QApplication.processEvents()
            logging.info(f"Similarity score: {sim_score}")
            
            try:
                # Use permutation test for significance
                p_val = self.permutation_p_value(y_ctrl, y_sample, sim_score, n_permutations=1000)
                logging.info(f"Permutation test p-value: {p_val}")
            
            except Exception:
                p_val = 1.0
                logging.info("Permutation test failed, p-value set to 1.0.")

            # Set p-value thresholds for correlational significance
            pval_threshold = 0.05
            logging.info(f"p-value threshold properly set to {pval_threshold}.")

            if time.time() - sim_start > 30:
                logging.warning("Similarity metrics computation is taking unusually long (possible infinite loop).")
            self.ui.remove_countdown_gif_and_timer()

            conotoxin_like = (p_val <= pval_threshold)
            return {
                "similarity_percent": sim_score,
                "p_value": p_val,
                "conotoxin_like": conotoxin_like
            }
        
        except Exception as e:
            self.ui.remove_countdown_gif_and_timer()
            logging.error(f"Error in compute_similarity_metrics: {e}")
            QMessageBox.critical(self.ui, "Error Computing Similarity", f"An error occurred while computing similarity metrics.\n\nError: {e}")
            self.ui.restore_cursor()
            self.rh.return_home_from_error()
            return {"similarity_percent": 0, "p_value": 1.0, "conotoxin_like": False}

    def analyze_all_samples(self):
        try:
            logging.info("Starting analyze_all_samples method.")
            QApplication.processEvents()
            results = {}
            ctrl_name = self.positive_control
            logging.info(f"Using positive control: {ctrl_name}")
            if ctrl_name not in self.processed:
                raise ValueError(f"Positive control '{ctrl_name}' not found in processed data.")
            x_ctrl, y_ctrl = self.processed[ctrl_name]
            regions_ctrl = self.segment_regions(x_ctrl, y_ctrl)
            if not all(k in regions_ctrl for k in ['NUV', 'VIS', 'NIR']):
                raise ValueError("One or more spectral regions are missing in the positive control.")
            logging.info("Segmented positive control regions.")
            y_nuv_dec = self.deconvolve_voigt(*regions_ctrl['NUV'])
            logging.info("Deconvolved NUV region for control.")
            y_nir_dec = self.deconvolve_voigt(*regions_ctrl['NIR'])
            logging.info("Deconvolved NIR region for control.")
            _, y_vis = regions_ctrl['VIS']
            y_ctrl_full = np.concatenate([y_nuv_dec, y_vis, y_nir_dec])
            logging.info("Constructed full control curve.");
            if hasattr(self.ui, 'progress_bar'):
                self.ui.progress_bar.setValue(60)
                self.ui.set_splash_text()
            QApplication.processEvents()
            total = len(self.sample_names)
            pvals = []
            metrics_list = []
            for idx, name in enumerate(self.sample_names):
                sample_start = time.time()
                logging.info(f"Analyzing sample {idx+1}/{total}: {name}")
                if name not in self.processed:
                    raise ValueError(f"Sample '{name}' not found in processed data.")
                x_sample, y_sample = self.processed[name]
                regions_sample = self.segment_regions(x_sample, y_sample)
                if not all(k in regions_sample for k in ['NUV', 'VIS', 'NIR']):
                    raise ValueError(f"One or more spectral regions are missing in sample '{name}'.")
                logging.info(f"Segmented regions for {name}")
                QApplication.processEvents()
                y_nuv_sample = self.deconvolve_voigt(*regions_sample['NUV'])
                logging.info(f"Deconvolved NUV for {name}")
                QApplication.processEvents()
                y_nir_sample = self.deconvolve_voigt(*regions_sample['NIR'])
                logging.info(f"Deconvolved NIR for {name}")
                QApplication.processEvents()
                _, y_vis_sample = regions_sample['VIS']
                y_sample_full = np.concatenate([y_nuv_sample, y_vis_sample, y_nir_sample])
                metrics = self.compute_similarity_metrics(y_ctrl_full, y_sample_full)
                logging.info(f"Computed similarity metrics for {name}")
                metrics_list.append(metrics)
                pvals.append(metrics['p_value'])
                if hasattr(self.ui, 'progress_bar'):
                    progress = 60 + int(40 * (idx+1) / total)
                    self.ui.progress_bar.setValue(progress)
                    self.ui.set_splash_text()
                QApplication.processEvents()
                elapsed = time.time() - sample_start
                if elapsed > 30:
                    logging.warning(f"Analysis for sample {name} is taking unusually long: {elapsed:.1f}s")
            logging.info("Applying FDR correction to p-values.")
            rejected, pvals_corrected = FDRUtils.fdr_correction(pvals, alpha=0.05, method='fdr_bh')
            for i, name in enumerate(self.sample_names):
                metrics = metrics_list[i]
                metrics['fdr_p_value'] = pvals_corrected[i]
                metrics['conotoxin_like_fdr'] = bool(rejected[i])
                results[name] = metrics
            logging.info("All samples analyzed. Returning results.")
            try:
                logging.info("Autosaving results to JSON file.")
                import os
                autosave_dir = os.path.dirname(auto_save_file)
                if not os.path.exists(autosave_dir):
                    os.makedirs(autosave_dir)
                autosaved_data = []
                if os.path.exists(auto_save_file):
                    try:
                        with open(auto_save_file, 'r') as f:
                            autosaved_data = json.load(f)
                            if not isinstance(autosaved_data, list):
                                autosaved_data = []
                    except Exception as read_err:
                        logging.warning(f"Could not read existing autosave file: {read_err}")
                        autosaved_data = []
                analysis_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "results": []
                }
                for sample, metrics in results.items():
                    entry = {
                        "date": datetime.datetime.now().isoformat(),
                        "sample": sample,
                        "similarity_percent": safe_number_for_json(metrics.get("similarity_percent", 0)),
                        "p_value": safe_number_for_json(metrics.get("p_value", 1)),
                        "fdr_p_value": safe_number_for_json(metrics.get("fdr_p_value", 1)),
                        "conotoxin_like": bool(metrics.get("conotoxin_like", False)),
                        "conotoxin_like_fdr": bool(metrics.get("conotoxin_like_fdr", False))
                    }
                    analysis_entry["results"].append(entry)
                autosaved_data.append(analysis_entry)
                with open(auto_save_file, 'w') as f:
                    json.dump(autosaved_data, f, indent=2)
                logging.info(f"Results autosaved to {auto_save_file}.")
            except Exception as autosave_err:
                logging.error(f"Failed to autosave results: {autosave_err}")
            return results
        except Exception as e:
            logging.error(f"Error in analyze_all_samples: {e}")
            QMessageBox.critical(self.ui, "Error Analyzing Samples", f"An error occurred while analyzing samples.\n\nError: {e}")
            self.ui.restore_cursor()
            self.rh.return_home_from_error()
            return {}

    def segment_regions(self, x, y):
        """
        Segments the spectrum into NUV, VIS, and NIR regions.
        Returns a dict: {'NUV': (x_nuv, y_nuv), ...}
        """
        regions = {
            'NUV': (200, 400),
            'VIS': (400, 700),
            'NIR': (700, 1000)
        }
        segmented = {}
        for region, (lower, upper) in regions.items():
            mask = (x >= lower) & (x <= upper)
            if not np.any(mask):
                continue
            segmented[region] = (x[mask], y[mask])
        return segmented

    def permutation_p_value(self, y_ctrl, y_sample, sim_score, n_permutations=1000, random_state=None):
        """
        Compute a permutation p-value for the observed sim_score.
        """
        rng = np.random.default_rng(random_state)
        count = 0

        for _ in range(n_permutations):
            permuted = rng.permutation(y_sample)
            
            try:
                cos_sim = cosine_similarity([y_ctrl], [permuted])[0, 0]
                pearson_corr, _ = pearsonr(y_ctrl, permuted)
                euclid_dist = np.linalg.norm(y_ctrl - permuted)
                auc_diff = np.abs(np.trapezoid(y_ctrl) - np.trapezoid(permuted)) / np.trapezoid(y_ctrl)
                pca = PCA(n_components=1)
                X = np.vstack([y_ctrl, permuted])
                X_scaled = StandardScaler().fit_transform(X)
                pca.fit(X_scaled)
                pca_score = np.abs(pca.components_[0][0] - pca.components_[0][1])
                
                # Compute PLS score for every permutation (no skipping)
                try:
                    pls = PLSRegression(n_components=2)
                    pls.fit(y_ctrl.reshape(-1, 1), permuted)
                    pls_score = pls.score(y_ctrl.reshape(-1, 1), permuted)
                except Exception:
                    pls_score = 0
                sim_metrics = np.array([
                    cos_sim,
                    (pearson_corr + 1) / 2,
                    1 / (1 + euclid_dist),
                    1 - auc_diff,
                    1 - pca_score,
                    pls_score if pls_score > 0 else 0
                ])
                perm_sim_score = np.clip(np.mean(sim_metrics), 0, 1) * 100
                if perm_sim_score >= sim_score:
                    count += 1
            except Exception:
                continue
        p_value = (count + 1) / (n_permutations + 1)
        return p_value
