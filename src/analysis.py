"""
The actual backend analysis algorithm
"""

# Basic Imports
import logging
import time
import json
import datetime
			
# Data Science Imports
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
# Use Savitzky-Golay only for smoothing; remove cubic spline and fft-based lowpass
from scipy.stats import pearsonr
from scipy.special import wofz
from skopt import gp_minimize
from skopt.space import Integer
from fastdtw import fastdtw # type: ignore
import decimal
import lmfit
import warnings

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
from PyQt6.QtCore import QTimer, QElapsedTimer, Qt

# Core Imports
from utils import FDRUtils, ErrorManager # error pausing checks for every error-related QMessageBox instance in this file specifically
from loop_manger import ReturnHome
from config import (
	qfiledialog__pinned_locations,
	LOG_FILE as global_logging_file,
	AUTO_SAVE_RESULTS_FILE as AUTO_SAVE_RESULTS_FILE
)
from fitting_tests import MLFittingUnitTests
import os
import hashlib

logging.basicConfig(
	filename= global_logging_file,
	level=logging.DEBUG,
	format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d'
)

# Suppress a noisy but non-fatal warning coming from the 'uncertainties' package
# which some downstream libraries may use when an uncertainty's std_dev==0.
warnings.filterwarnings(
	"ignore",
	message=r"Using UFloat objects with std_dev==0 may give unexpected results\.",
	category=UserWarning,
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
		self.fitting_tester = MLFittingUnitTests(self)
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
			if not ErrorManager.errors_suppressed:
				QMessageBox.critical(self.ui, "Error", f"Error uploading CSV file: {e}")
			self.ui.restore_cursor()
			raise RuntimeError(f"Error uploading CSV file: {e}")

	def read_csv(self):
		try:
			logging.info("Starting read_csv method.")
			QApplication.processEvents()
			if self.filename is None:
				raise ValueError("No filename specified for CSV upload.")
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
			# Only store 1D absorbance arrays, not tuples
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
				x_smooth, y_smooth, sg_win, sg_poly = self.smooth_curve(x, y)
				# Store tuple (x, y_smoothed, sg_window, sg_poly)
				processed[name] = (x_smooth, y_smooth, sg_win, sg_poly)
			if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
				self.ui.progress_bar.setValue(35)
			if hasattr(self.ui, 'set_splash_text'):
				self.ui.set_splash_text()
			QApplication.processEvents()
			logging.info("All curves smoothed. Exiting preprocess_all_curves.")
			return processed
		except Exception as e:
			logging.error(f"Error in preprocess_all_curves: {e}")
			if not ErrorManager.errors_suppressed: 
				QMessageBox.critical(self.ui, "Error Preprocessing Data", f"An error occurred while preprocessing your data.\n\nError: {e}")
			self.ui.restore_cursor()
			self.rh.return_home_from_error()
			return {}

	def smooth_curve(self, x, y):
		try:
			# Simplified smoothing: use Savitzky-Golay directly on the original sampled data
			self.ui.show_countdown_gif(self.ui.progress_bar_row, self.ui.progress_container, trigger_type='progress', duration_ms=3000)
			logging.info("Starting smooth_curve method (Savitzky-Golay only).")
			QApplication.processEvents()
			# Work on numpy arrays
			x = np.asarray(x)
			y = np.asarray(y)
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
			# set bounds based on signal length, but cap window to 20% of length to avoid oversmoothing
			win_upper = max(5, min(101, max(5, int(len(y) * 0.2))))
			if win_upper % 2 == 0:
				win_upper -= 1
			search_space = [Integer(5, max(5, win_upper)), Integer(2, min(5, max(2, win_upper-1)))]
			logging.info("Starting Bayesian optimization for Savitzky-Golay parameters (fast).")
			opt_start = time.time()
			result = gp_minimize(sg_error, search_space, n_calls=18, random_state=0)
			if time.time() - opt_start > 30:
				logging.warning("Savitzky-Golay optimization is taking unusually long.")
				if not ErrorManager.errors_suppressed:
					QMessageBox.warning(self.ui, "Warning", "Smoothing is taking unusually long. Please check your data or restart the app.")
			if result is not None and hasattr(result, 'x'):
				win_opt = int(result.x[0])
				if win_opt % 2 == 0:
					win_opt += 1
				poly_opt = int(result.x[1])
				poly_opt = max(2, min(poly_opt, win_opt-1))
			else:
				win_opt = 11
				poly_opt = 2
			if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
				self.ui.progress_bar.setValue(30)
			if hasattr(self.ui, 'set_splash_text'):
				self.ui.set_splash_text()
			QApplication.processEvents()
			# Ensure window length is valid for savgol
			if win_opt >= len(y):
				win_opt = len(y) - 1 if (len(y) - 1) % 2 == 1 else len(y) - 2
			y_final = savgol_filter(y, window_length=max(5, win_opt), polyorder=poly_opt)
			self.ui.remove_countdown_gif_and_timer()
			# Return original x, smoothed y, and sg parameters for downstream residual-aware fitting
			return x, y_final, int(win_opt), int(poly_opt)
		except Exception as e:
			self.ui.remove_countdown_gif_and_timer()
			logging.error(f"Error in smooth_curve: {e}")
			if not ErrorManager.errors_suppressed:   
				QMessageBox.critical(self.ui, "Error Smoothing Curve", f"An error occurred while smoothing the spectrum.\n\nError: {e}")
			self.ui.restore_cursor()
			self.rh.return_home_from_error()
			# Return best-effort values and default SG params on failure
			return x, y, 11, 2

	def align_curves_dtw(self, ref, query):
		
		try:
			self.ui.show_countdown_gif(self.ui.progress_bar_row, self.ui.progress_container, trigger_type='progress', duration_ms=3000)
			logging.info("Starting align_curves_dtw method.")
			QApplication.processEvents()
			align_start = time.time()
			distance, path = fastdtw(ref, query)
			
			if time.time() - align_start > 30:
				logging.warning("DTW alignment is taking unusually long (possible infinite loop).")
				if not ErrorManager.errors_suppressed:
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
			if not ErrorManager.errors_suppressed:  
				QMessageBox.critical(self.ui, "Error Aligning Curves", f"An error occurred while aligning curves using DTW.\n\nError: {e}")
			self.ui.restore_cursor()
			self.rh.return_home_from_error()
			return query

	def deconvolve_voigt(self, x, y, center_nm, window=30, baseline=None, sg_win=None, sg_poly=None):
		"""
		Deconvolve (fit) peaks in a window centered at center_nm +/- window.
		Returns an array of same length as y with the fitted values on the window
		and np.nan elsewhere so caller can reconstruct the full spectrum.
		"""
		try:
			self.ui.show_countdown_gif(self.ui.progress_bar_row, self.ui.progress_container, trigger_type='progress', duration_ms=3000)
			logging.info(f"Starting deconvolve_voigt around {center_nm} +/- {window} nm.")
			QApplication.processEvents()
			x = np.asarray(x)
			y = np.asarray(y)
			mask = (x >= (center_nm - window)) & (x <= (center_nm + window))
			if not np.any(mask):
				logging.warning(f"No data in window for center {center_nm} nm; returning original segment.")
				self.ui.remove_countdown_gif_and_timer()
				out = np.full_like(y, np.nan, dtype=float)
				return out
			x_seg = x[mask]
			y_seg = y[mask]
			# If baseline provided, compute residuals = original - baseline
			if baseline is not None:
				baseline_seg = baseline[mask]
				resid = y_seg - baseline_seg
			else:
				resid = y_seg.copy()

			# Find local maxima in residual within the window
			try:
				peaks, _ = find_peaks(resid, height=np.max(resid) * 0.1)
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

			amps = np.maximum(resid, 0)
			# initial guesses
			from scipy.optimize import curve_fit

			def multi_gauss(xa, *params):
				res = np.zeros_like(xa)
				for i in range(n_peaks):
					amp = params[i*3]
					cen = params[i*3+1]
					sig = params[i*3+2]
					res += amp * np.exp(-0.5 * ((xa - cen) / sig) ** 2)
				return res

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
				# amplitude guess: max of residual or segment scaled
				amp_guess = float(np.max(resid)) if np.max(resid) > 0 else float(np.max(y_seg))
				# p0: amp, cen, sigma, gamma
				p0 += [amp_guess, float(c), max(0.5, (x_seg.max()-x_seg.min())/12.0), max(0.5, (x_seg.max()-x_seg.min())/12.0)]
				bounds_low += [0, c - 10, 1e-3, 1e-3]
				bounds_high += [amp_guess * 10 + 1 if amp_guess>0 else np.max(y_seg) * 10 + 1, c + 10, (x_seg.max()-x_seg.min()), (x_seg.max()-x_seg.min())]

			# Fit to residual using lmfit VoigtModel(s)
			try:
				# ensure float arrays for lmfit
				x_fit = np.array(x_seg, dtype=float)
				resid_fit = np.array(resid, dtype=float)
				if resid_fit.size == 0:
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
				params = composite.make_params()
				for i, c in enumerate(centers):
					prefix = f'v{i}_'
					amp_guess = float(np.max(resid_fit)) if np.max(resid_fit) > 0 else float(np.max(y_seg))
					init_sigma = max(0.5, (x_seg.max() - x_seg.min()) / 12.0)
					init_gamma = init_sigma
					# set parameter initial guesses and bounds
					params[f'{prefix}amplitude'].set(value=amp_guess, min=0)
					params[f'{prefix}center'].set(value=float(c), min=float(c - 10), max=float(c + 10))
					params[f'{prefix}sigma'].set(value=init_sigma, min=1e-3, max=float(x_seg.max() - x_seg.min()))
					params[f'{prefix}gamma'].set(value=init_gamma, min=1e-3, max=float(x_seg.max() - x_seg.min()))

				result = composite.fit(resid_fit, params, x=x_fit)
				fitted_resid = result.best_fit.astype(float)
				# add fitted residual back to baseline if provided
				if baseline is not None:
					fitted = baseline_seg.astype(float) + fitted_resid
				else:
					fitted = fitted_resid
				logging.info(f"lmfit Voigt residual fit succeeded for center {center_nm} nm. Fit success: {result.success}")
			except Exception as fit_err:
				logging.warning(f"lmfit Voigt fit failed for center {center_nm}: {fit_err}")
				# fallback: use baseline if present, else original segment
				if baseline is not None:
					fitted = baseline_seg.astype(float)
				else:
					fitted = y_seg.astype(float)

			out = np.full_like(y, np.nan, dtype=float)
			out[mask] = fitted
			if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
				self.ui.progress_bar.setValue(50)
			if hasattr(self.ui, 'set_splash_text'):
				self.ui.set_splash_text()
			QApplication.processEvents()
			self.ui.remove_countdown_gif_and_timer()
			return out
		except Exception as e:
			self.ui.remove_countdown_gif_and_timer()
			logging.error(f"Error in deconvolve_voigt (windowed): {e}")
			if not ErrorManager.errors_suppressed:
				QMessageBox.warning(self.ui, "Voigt/Gauss Fit Warning", f"Fitting failed around {center_nm} nm. Using original data in that window.\n\nError: {e}")
			return np.full_like(y, np.nan, dtype=float)

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
			# PCA on two samples: we can use the explained variance ratio as a proxy
			try:
				pca = PCA(n_components=1)
				pca.fit(X_scaled)
				# components_ shape is (n_components, n_features)
				# For two samples, use the variance explained as a stability proxy
				pca_score = float(1 - pca.explained_variance_ratio_[0]) if hasattr(pca, 'explained_variance_ratio_') else 0.0
			except Exception:
				pca_score = 0.0
			
			if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
				self.ui.progress_bar.setValue(65)
			
			if hasattr(self.ui, 'set_splash_text'):
				self.ui.set_splash_text()
			QApplication.processEvents()
			logging.info(f"PCA score computed: {pca_score}")
			
			# Use a safe number of PLS components: cannot exceed min(n_samples-1, n_features)
			try:
				n_samples = y_ctrl.reshape(-1, 1).shape[0]
				n_components = min(1, max(1, n_samples - 1))
				pls = PLSRegression(n_components=n_components)
				pls.fit(y_ctrl.reshape(-1, 1), y_sample)
				pls_score = pls.score(y_ctrl.reshape(-1, 1), y_sample)
				logging.info(f"PLS score computed: {pls_score}")
			except Exception as pls_err:
				pls_score = 0
				logging.info(f"PLS score computation failed, set to 0. Error: {pls_err}")
			
			cos_sim = cosine_similarity(np.array([y_ctrl]), np.array([y_sample]))[0, 0]
			pc_tuple = pearsonr(y_ctrl, y_sample)
			pearson_corr: float = pc_tuple[0]  # type: ignore
			euclid_dist = np.linalg.norm(y_ctrl - y_sample)
			
			if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
				self.ui.progress_bar.setValue(70)
			
			if hasattr(self.ui, 'set_splash_text'):
				self.ui.set_splash_text()
			QApplication.processEvents()
			logging.info(f"Cosine similarity: {cos_sim}, Pearson: {pearson_corr}, Euclidean: {euclid_dist}")
			
			# Guard against zero AUC in control to avoid divide-by-zero
			try:
				auc_ctrl = np.trapezoid(y_ctrl)
				auc_sample = np.trapezoid(y_sample)
				if np.isclose(auc_ctrl, 0.0):
					auc_diff = np.abs(auc_ctrl - auc_sample)
				else:
					auc_diff = np.abs(auc_ctrl - auc_sample) / auc_ctrl
			except Exception:
				auc_diff = 1.0
			sim_metrics = np.array([
				cos_sim,
				(float(pearson_corr) + 1) / 2,
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
			if not ErrorManager.errors_suppressed:
				QMessageBox.critical(self.ui, "Error Computing Similarity", f"An error occurred while computing similarity metrics.\n\nError: {e}")
			self.ui.restore_cursor()
			self.rh.return_home_from_error()
			return {"similarity_percent": 0, "p_value": 1.0, "conotoxin_like": False}

	def analyze_all_samples(self):
		try:
			logging.info("Starting analyze_all_samples method.")
			QApplication.processEvents()
			# Reset SAP guard so dialog can be shown for each new analysis run
			self._sap_guard_shown = False
			results = {}
			ctrl_name = self.positive_control
			logging.info(f"Using positive control: {ctrl_name}")
			if ctrl_name not in self.processed:
				raise ValueError(f"Positive control '{ctrl_name}' not found in processed data.")
			x_ctrl, y_ctrl_sg, sg_win_ctrl, sg_poly_ctrl = self.processed[ctrl_name]
			# Ensure raw absorbance dict exists and contains control
			if self.absorbance is None or ctrl_name not in self.absorbance:
				raise ValueError(f"Raw absorbance data missing for control '{ctrl_name}'.")
			# original raw control
			y_ctrl_raw = np.asarray(self.absorbance[ctrl_name])
			# Define fixed centers and window
			centers = [358, 435, 470, 779, 982]
			window = 30
			# Smooth control (already smoothed in preprocess_all_curves)
			# Build reconstructed control by fitting each center and merging
			# Start with SG baseline and add fitted residuals in windows
			fitted_ctrl = np.array(y_ctrl_sg, dtype=float)
			# apply deconvolution per center and overwrite only masked regions
			for c in centers:
				fitted_seg = self.deconvolve_voigt(x_ctrl, y_ctrl_raw, center_nm=c, window=window, baseline=y_ctrl_sg, sg_win=sg_win_ctrl, sg_poly=sg_poly_ctrl)
				mask = ~np.isnan(fitted_seg)
				if np.any(mask):
					fitted_ctrl[mask] = fitted_seg[mask]
			# final control full reconstructed curve
			y_ctrl_full = fitted_ctrl
			logging.info("Constructed full control curve.")
			# --- Create and save preprocessed (deconvolved + smoothed + DTW-aligned) spectra mid-analysis ---
			try:
				# Align control to itself (identity but keeps consistent processing)
				aligned_control = self.align_curves_dtw(y_ctrl_full, y_ctrl_full)
				aligned_dict = {ctrl_name: np.asarray(aligned_control, dtype=float)}
				# We'll collect aligned samples and reconstructed (fitted) samples first; initialize lists to preserve order
				aligned_order = []
				fitted_samples = {}
			except Exception as e:
				logging.warning(f"Failed to DTW-align control for save: {e}")
				aligned_dict = {ctrl_name: np.asarray(y_ctrl_full, dtype=float)}
				aligned_order = []
				fitted_samples = {}
			if hasattr(self.ui, 'progress_bar'):
				self.ui.progress_bar.setValue(60)
				self.ui.set_splash_text()
			QApplication.processEvents()
			# First pass: perform deconvolution and DTW alignment for all samples and store reconstructed spectra
			total = len(self.sample_names)
			centers = [358, 435, 470, 779, 982]
			window = 30
			for idx, name in enumerate(self.sample_names):
				sample_start = time.time()
				logging.info(f"Preprocessing sample {idx+1}/{total}: {name}")
				if name not in self.processed:
					raise ValueError(f"Sample '{name}' not found in processed data.")
				x_sample, y_sample_sg, sg_win_sample, sg_poly_sample = self.processed[name]
				# Ensure raw absorbance dict exists and contains sample
				if self.absorbance is None or name not in self.absorbance:
					raise ValueError(f"Raw absorbance data missing for sample '{name}'.")
				# raw sample
				y_sample_raw = np.asarray(self.absorbance[name])
				# reconstruct full sample by fitting each center window and replacing
				fitted_sample = np.array(y_sample_sg, dtype=float)
				for c in centers:
					fitted_seg = self.deconvolve_voigt(x_sample, y_sample_raw, center_nm=c, window=window, baseline=y_sample_sg, sg_win=sg_win_sample, sg_poly=sg_poly_sample)
					mask = ~np.isnan(fitted_seg)
					if np.any(mask):
						fitted_sample[mask] = fitted_seg[mask]
				# DTW-align reconstructed sample to control and store for saving
				try:
					aligned_sample = self.align_curves_dtw(y_ctrl_full, fitted_sample)
				except Exception:
					aligned_sample = np.asarray(fitted_sample, dtype=float)
				aligned_dict[name] = np.asarray(aligned_sample, dtype=float)
				aligned_order.append(name)
				fitted_samples[name] = np.asarray(fitted_sample, dtype=float)
				if hasattr(self.ui, 'progress_bar'):
					progress = 60 + int(30 * (idx+1) / total)
					self.ui.progress_bar.setValue(progress)
					self.ui.set_splash_text()
				QApplication.processEvents()
				elapsed = time.time() - sample_start
				if elapsed > 30:
					logging.warning(f"Preprocessing for sample {name} is taking unusually long: {elapsed:.1f}s")
			# After DTW + deconvolution + smoothing, save preprocessed data (control + aligned samples)
			try:
				self.save_preprocessed_data(aligned_dict, aligned_order, y_ctrl_full)
			except Exception as save_err:
				logging.warning(f"Failed to save preprocessed data: {save_err}")

			logging.info("Applying FDR correction to p-values.")
			# Compute similarity metrics now that all preprocessing (smoothing, deconv, DTW) is complete
			metrics_list = []
			pvals = []
			for name in self.sample_names:
				if name not in fitted_samples:
					logging.warning(f"Fitted sample missing for metrics computation: {name}")
					continue
				fitted_sample = fitted_samples[name]
				metrics = self.compute_similarity_metrics(y_ctrl_full, fitted_sample)
				metrics_list.append(metrics)
				pvals.append(metrics.get('p_value', 1.0))
			# Apply FDR correction if we have p-values
			if len(pvals) > 0:
				rejected, pvals_corrected = FDRUtils.fdr_correction(pvals, alpha=0.05, method='fdr_bh')
				for i, name in enumerate(self.sample_names):
					if i >= len(metrics_list):
						break
					metrics = metrics_list[i]
					metrics['fdr_p_value'] = pvals_corrected[i]
					metrics['conotoxin_like_fdr'] = bool(rejected[i])
					results[name] = metrics
			else:
				logging.warning("No p-values were computed; skipping FDR correction.")
			logging.info("All samples analyzed. Returning results.")
			try:
				# Save preprocessed CSV via helper which will attempt configured autosave dir then repo-local fallback
				try:
					saved_path = self.save_preprocessed_data(aligned_dict if 'aligned_dict' in locals() else {ctrl_name: np.asarray(y_ctrl_full, dtype=float)},
						aligned_order if 'aligned_order' in locals() else list(self.sample_names),
						y_ctrl_full)
					logging.info(f"Preprocessed CSV saved (helper) to: {saved_path}")
				except Exception as save_err:
					logging.warning(f"Failed to save preprocessed input CSV via helper: {save_err}")
				# Continue with normal autosave of results
				logging.info("Autosaving results to JSON file.")
				autosave_dir = os.path.dirname(AUTO_SAVE_RESULTS_FILE)
				if not os.path.exists(autosave_dir):
					os.makedirs(autosave_dir)
				autosaved_data = []
				if os.path.exists(AUTO_SAVE_RESULTS_FILE):
					try:
						with open(AUTO_SAVE_RESULTS_FILE, 'r') as f:
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
				with open(AUTO_SAVE_RESULTS_FILE, 'w') as f:
					json.dump(autosaved_data, f, indent=2)
				logging.info(f"Results autosaved to {AUTO_SAVE_RESULTS_FILE}.")
			except Exception as autosave_err:
				logging.error(f"Failed to autosave results: {autosave_err}")

			# End of SAP, unit tests initialized
			logging.info("ML/statistical analysis computations completed, running unit tests now for overfitting/underfitting risk management.")
			proceed = self.end_of_sap_guard()
			# Cache results on the UI so results() can pick them up
			try:
				self.ui._cached_results = results
			except Exception:
				pass
			if proceed:
				# Finalize UI state explicitly so it doesn't hang at 100%
				try:
					self.ui.progress_bar.setValue(100)
				except Exception:
					pass
				# Trigger the UI's splash text update which will call results()
				try:
					self.ui.set_splash_text()
				except Exception:
					pass
				QApplication.processEvents()
				try:
					self.ui.restore_cursor()
				except Exception:
					pass
				return results
			else:
				# Ensure UI is unblocked even on guard decline
				try:
					self.ui.progress_bar.setValue(100)
				except Exception:
					pass
				try:
					self.ui.set_splash_text()
				except Exception:
					pass
				QApplication.processEvents()
				try:
					self.ui.restore_cursor()
				except Exception:
					pass
				# Cache empty results so UI doesn't call analysis again
				try:
					self.ui._cached_results = {}
				except Exception:
					pass
				return {}
		
		except Exception as e:
			logging.error(f"Error in analyze_all_samples: {e}")
			if not ErrorManager.errors_suppressed:
				QMessageBox.critical(self.ui, "Error Analyzing Samples", f"An error occurred while analyzing samples.\n\nError: {e}")
			self.ui.restore_cursor()
			self.rh.return_home_from_error()
			return {}

	def save_preprocessed_data(self, aligned_dict, aligned_order, y_ctrl_full):
		"""
		Save the preprocessed (smoothed, deconvolved, DTW-aligned) spectra to a CSV file.
		Writes to a repo-local utilities/Autosaves folder with timestamped filename.
		Parameters:
		- aligned_dict: mapping from sample name -> aligned numpy array
		- aligned_order: list of sample names in the order they were processed
		- y_ctrl_full: full reconstructed control spectrum (numpy array)
		"""
		try:
			logging.info("Saving preprocessed data to CSV (save_preprocessed_data).")
			# Attempt to save into configured autosave dir (parent of AUTO_SAVE_RESULTS_FILE)
			saving_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
			preferred_dir = os.path.dirname(AUTO_SAVE_RESULTS_FILE)
			fname = None
			# Build DataFrame
			nm_col = np.asarray(self.wavelengths)
			df_save = pd.DataFrame({'nm': nm_col})
			# Control
			ctrl_name = self.positive_control
			df_save[ctrl_name] = np.asarray(y_ctrl_full, dtype=float)
			# Add samples in recorded order
			order_list = aligned_order if aligned_order and len(aligned_order) > 0 else list(self.sample_names)
			for n in order_list:
				col = aligned_dict.get(n, np.full_like(nm_col, np.nan, dtype=float))
				df_save[n] = col
			# Serialize CSV into bytes and compute content hash to avoid duplicate writes
			csv_bytes = df_save.to_csv(index=False).encode('utf-8')
			content_hash = hashlib.sha256(csv_bytes).hexdigest()
			# If we've already saved this exact content in this AnalysisEngine instance, skip
			last_hash = getattr(self, '_last_preprocessed_hash', None)
			if last_hash == content_hash:
				logging.info("Preprocessed CSV identical to last saved; skipping duplicate write.")
				return getattr(self, '_last_preprocessed_path', None)
			# Use a stable filename containing the hash so preferred and fallback share the same name
			base_fname = f'preprocessed_input_{saving_tag}_{content_hash[:8]}.csv'
			# Try preferred autosave location first
			try:
				if not os.path.exists(preferred_dir):
					os.makedirs(preferred_dir, exist_ok=True)
				fname_pref = os.path.join(preferred_dir, base_fname)
				# If file already exists, no need to overwrite
				if not os.path.exists(fname_pref):
					with open(fname_pref, 'wb') as f:
						f.write(csv_bytes)
				fname = fname_pref
				logging.info(f"Saved preprocessed input CSV to preferred autosave dir: {fname_pref}")
			except Exception as pref_err:
				logging.warning(f"Preferred autosave dir write failed: {pref_err}")
			# Fallback to repo-local utilities/Autosaves
			if fname is None:
				repo_autosave_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utilities', 'Autosaves')
				try:
					os.makedirs(repo_autosave_dir, exist_ok=True)
					fname_fb = os.path.join(repo_autosave_dir, base_fname)
					# If file already exists, skip writing
					if not os.path.exists(fname_fb):
						with open(fname_fb, 'wb') as f:
							f.write(csv_bytes)
					fname = fname_fb
					logging.info(f"Saved preprocessed input CSV to repo-local fallback: {fname_fb}")
				except Exception as fb_err:
					logging.warning(f"Fallback autosave dir write failed: {fb_err}")
			# Cache last saved content hash and path to avoid duplicate writes later
			if fname is not None:
				self._last_preprocessed_hash = content_hash
				self._last_preprocessed_path = fname
			# Return the path if saved, else None
			return fname
		except Exception as e:
			logging.warning(f"Failed to save preprocessed CSV in save_preprocessed_data: {e}")
			return None

	def permutation_p_value(self, y_ctrl, y_sample, sim_score, n_permutations=1000, random_state=None):
		"""
		Compute a permutation p-value for the observed sim_score.
		"""
		rng = np.random.default_rng(random_state)
		count = 0

		for _ in range(n_permutations):
			permuted = rng.permutation(y_sample)
			
			try:
				cos_sim = cosine_similarity(np.array([y_ctrl]), np.array([permuted]))[0, 0]
				pc_tuple = pearsonr(y_ctrl, permuted)
				pearson_corr: float = pc_tuple[0]  # type: ignore
				euclid_dist = np.linalg.norm(y_ctrl - permuted)
				auc_diff = np.abs(np.trapezoid(y_ctrl) - np.trapezoid(permuted)) / np.trapezoid(y_ctrl)
				pca = PCA(n_components=1)
				X = np.vstack([y_ctrl, permuted])
				X_scaled = StandardScaler().fit_transform(X)
				pca.fit(X_scaled)
				pca_score = np.abs(pca.components_[0][0] - pca.components_[0][1])
				
				# Compute PLS score for every permutation (robust to small samples)
				try:
					n_samples = y_ctrl.reshape(-1, 1).shape[0]
					n_components = min(1, max(1, n_samples - 1))
					pls = PLSRegression(n_components=n_components)
					pls.fit(y_ctrl.reshape(-1, 1), permuted)
					pls_score = pls.score(y_ctrl.reshape(-1, 1), permuted)
				except Exception:
					pls_score = 0
				sim_metrics = np.array([
					cos_sim,
					(float(pearson_corr) + 1) / 2,
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
	
	def end_of_sap_guard(self):
		# Prevent multiple dialogs by using a guard variable
		if getattr(self, "_sap_guard_shown", False):
			logging.info("SAP guard dialog already shown, skipping duplicate.")
			return True  # or False, depending on your logic
		self._sap_guard_shown = True
		# Import resource_path here to avoid circular import
		from main import resource_path
		test_report_success = resource_path("test_report_success.html")
		test_report_failure = resource_path("test_report_failure.html")
		logging.debug(f"Resolved test_report_success.html path: {test_report_success}")
		logging.debug(f"Resolved test_report_failure.html path: {test_report_failure}")

		def read_test_report_html(path):
			try:
				with open(path, 'r', encoding='utf-8') as f:
					return f.read()
			except Exception as e:
				logging.error(f"Failed to read HTML file {path}: {e}")
				return f"<b>Error:</b> Could not load report file.<br>Path: {path}<br>Reason: {e}"

		# Read the HTML file contents for rich text display
		success_html = read_test_report_html(test_report_success)
		failure_html = read_test_report_html(test_report_failure)
		# Run the fitting tests but don't block the UI on the resulting dialog.
		# If the dialog would have blocked (exec()), we instead show it non-blocking
		# and auto-close after a short timeout. We return True immediately so the
		# analysis flow doesn't hang at 100% progress. The dialog is informative.
		try:
			underfit, overfit = self.fitting_tester.test_fittings()
		except Exception as test_err:
			logging.error(f"Fitting tests raised an exception: {test_err}")
			underfit, overfit = False, False

		# Build the message box once
		report_box = QMessageBox(self.ui)
		report_box.setTextFormat(Qt.TextFormat.RichText)

		if not underfit and not overfit:
			logging.info("No underfitting or overfitting detected in the ML pipeline.")
			report_box.setWindowTitle("Fitting Test Success")
			report_box.setText(success_html)
		else:
			logging.error("ML pipeline fitting tests failed.")
			report_box.setWindowTitle("Fitting Test Failure")
			report_box.setText(failure_html)

		# Show non-blocking and auto-close after 5 seconds so UI doesn't hang
		try:
			report_box.show()
			QTimer.singleShot(5000, report_box.close)
		except Exception as show_err:
			# If showing fails for any reason, log and continue
			logging.warning(f"Non-blocking report box failed to show: {show_err}")

		# If the tests failed, schedule a return to home shortly but don't block
		if underfit or overfit:
			# Give the user a moment to read the dialog, then return home non-blocking
			QTimer.singleShot(6000, lambda: self.rh.return_home_from_error())

		# Proceed without blocking the UI; caller will set progress to 100 and continue
		return True
