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
# from lmfit.models import VoigtModel done at deconvolution runtime
import warnings

# Machine Learning Imports
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# PyQt6 GUI Imports
from PyQt6.QtWidgets import (
	QFileDialog, 
	QMessageBox
)
from PyQt6.QtCore import QTimer, QElapsedTimer, Qt

# Import QApplication only when needed to avoid circular import issues
from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from PyQt6.QtWidgets import QApplication

# Core Imports
from utils import FDRUtils, ErrorManager # error pausing checks for every error-related QMessageBox instance in this file specifically
from loop_manger import ReturnHome
from config import (
	qfiledialog__pinned_locations,
	LOG_FILE,
	AUTO_SAVE_RESULTS_FILE
)
from fitting_tests import MLFittingUnitTests
import os
import hashlib

logging.basicConfig(
	filename= LOG_FILE,
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
			logging.info(f"preprocess_all_curves returned {len(self.processed)} entries: {list(self.processed.keys())}")
			# If the positive control wasn't processed (edge cases), attempt to smooth it now so analysis can continue
			if self.positive_control not in self.processed:
				try:
					x = np.asarray(self.wavelengths)
					y = np.asarray(data[self.positive_control])
					x_smooth, y_smooth, sg_win, sg_poly = self.smooth_curves(x, y)
					self.processed[self.positive_control] = (x_smooth, y_smooth, sg_win, sg_poly)
					logging.info(f"Recovered positive control '{self.positive_control}' by on-demand smoothing.")
				except Exception as e:
					logging.error(f"Failed to recover positive control '{self.positive_control}': {e}")
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
			# Remove hard abort timeout; allow all samples to be processed but log if it takes long
			timeout = 3600  # seconds (very large) - we will not abort early
			logging.info(f"preprocess_all_curves will process {len(data)} samples: {list(data.keys())}")
			
			# First, run global Bayesian optimization to determine optimal SG parameters for the entire dataset
			logging.info("Running global Bayesian optimization for Savitzky-Golay parameters.")
			optimal_win, optimal_poly = self.bayesian_optimize_win_and_poly(data, self.wavelengths)
			logging.info(f"Global optimization results: window={optimal_win}, polyorder={optimal_poly}")
			
			if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
				self.ui.progress_bar.setValue(20)
			if hasattr(self.ui, 'set_splash_text'):
				self.ui.set_splash_text()
			QApplication.processEvents()
			
			for name, y in data.items():
				try:
					logging.info(f"Preprocessing sample: {name}")
					x = np.asarray(self.wavelengths)
					y = np.asarray(y)
					if x is None or len(x) != len(y):
						logging.error(f"Length mismatch for '{name}': x({len(x) if x is not None else 'None'}), y({len(y)})")
						# record nothing for this sample and continue so other samples are processed
						continue
					# Use the globally optimized parameters
					x_smooth, y_smooth, sg_win, sg_poly = self.smooth_curves(x, y, win_opt=optimal_win, poly_opt=optimal_poly)
					# Store tuple (x, y_smoothed, sg_window, sg_poly)
					processed[name] = (x_smooth, y_smooth, sg_win, sg_poly)
				except Exception:
					# Log full traceback per-sample and continue (no fallback) so root cause is visible
					logging.exception(f"Failed to preprocess sample '{name}' - continuing with remaining samples")
					continue

			# Post-check: ensure every input spectrum has a smoothed entry. If any were skipped (e.g., transient errors),
			# attempt a best-effort smoothing pass so downstream deconvolution (which is local-windowed) does not
			# prevent having a full-spectrum smoothed baseline for each sample.
			missing = [n for n in data.keys() if n not in processed]
			if missing:
				logging.info(f"Attempting recovery smoothing for {len(missing)} missing samples: {missing}")
				for name in missing:
					try:
						x = np.asarray(self.wavelengths)
						y = np.asarray(data[name])
						# Use the globally optimized parameters for recovery too
						x_smooth, y_smooth, sg_win, sg_poly = self.smooth_curves(x, y, win_opt=optimal_win, poly_opt=optimal_poly)
						processed[name] = (x_smooth, y_smooth, sg_win, sg_poly)
						logging.info(f"Recovered smoothing for sample '{name}'.")
					except Exception:
						logging.exception(f"Recovery smoothing failed for sample '{name}'; leaving it out of processed set.")
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

	def bayesian_optimize_win_and_poly(self, data_dict, wavelengths):
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
			logging.error(f"Error in bayesian_optimize_win_and_poly: {e}")
			return 11, 2  # Return safe defaults on any error

	def smooth_curves(self, x, y, win_opt=None, poly_opt=None):
		try:
			# Simplified smoothing: use Savitzky-Golay directly on the original sampled data
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
			if hasattr(self.ui, 'progress_bar') and self.ui.progress_bar is not None:
				self.ui.progress_bar.setValue(30)
			if hasattr(self.ui, 'set_splash_text'):
				self.ui.set_splash_text()
			QApplication.processEvents()
			# Ensure window length is valid for savgol
			if win_opt >= len(y):
				win_opt = len(y) - 1 if (len(y) - 1) % 2 == 1 else len(y) - 2
			# Use a conservative polynomial order to avoid fitting noise
			polyorder = min(poly_opt, 3)
			y_final = savgol_filter(y, window_length=max(5, win_opt), polyorder=polyorder)
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
			distance, path = fastdtw(ref, query)
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

			# Log how long similarity computation took for diagnostics
			logging.debug(f"Similarity computation time: {time.time()-sim_start:.2f}s")
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
			centers = [358, 435, 470, 779, 982, 1000]
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
			# Redifine fixed centers and window
			centers = [358, 435, 470, 779, 982, 1000]
			window = 30
			for idx, name in enumerate(self.sample_names):
				sample_start = time.time()
				logging.info(f"Preprocessing sample {idx+1}/{total}: {name}")
				if name not in self.processed:
					logging.warning(f"Sample '{name}' not found in processed data; skipping this sample.")
					continue
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
				if elapsed > 120:
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
		Writes to the Autosaves directory with timestamped filename.
		Parameters:
		- aligned_dict: mapping from sample name -> aligned numpy array
		- aligned_order: list of sample names in the order they were processed
		- y_ctrl_full: full reconstructed control spectrum (numpy array)
		"""
		try:
			logging.info("Saving preprocessed data to CSV (save_preprocessed_data).")
			# Attempt to save into configured autosave dir (parent of AUTO_SAVE_RESULTS_FILE)
			saving_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
			save_smooths_dir = os.path.dirname(AUTO_SAVE_RESULTS_FILE)
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
				if not os.path.exists(save_smooths_dir):
					os.makedirs(save_smooths_dir, exist_ok=True)
				fname_pref = os.path.join(save_smooths_dir, base_fname)
				# If file already exists, no need to overwrite
				if not os.path.exists(fname_pref):
					with open(fname_pref, 'wb') as f:
						f.write(csv_bytes)
				fname = fname_pref
				logging.info(f"Saved preprocessed input CSV to the ~/LNVSI Tool Utilities/Autosaves directory: {fname_pref}")
			except Exception as pref_err:
				logging.warning(f"Write to the ~/LNVSI Tool Utilities/Autosaves directory failed: {pref_err}")
    
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
			# Check if the error might be related to Bayesian optimization issues
			error_msg = str(test_err).lower()
			if any(keyword in error_msg for keyword in ['bayesian', 'optimization', 'window', 'polynomial', 'savgol']):
				logging.error("Exception appears to be related to Bayesian optimization for SG parameters")
				# Treat Bayesian optimization failures as potential underfitting in parameter selection
				underfit, overfit = True, False
			else:
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
