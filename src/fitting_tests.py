
import numpy as np
import logging

class MLFittingUnitTests:
	"""
	Class for comprehensive overfitting/underfitting unit tests for all stages of the analysis pipeline.
	Interacts with an AnalysisEngine instance via self.analysis.
	"""
	def __init__(self, analysis):
		self.analysis = analysis

	def test_fittings(self):
		logging.info("Fitting tests initialized")  # Confirm function entry
		underfitting = False
		overfitting = False
		log_msgs = []

		# --- Pre-ML: Curve fitting checks (lmfit) ---
		# If fit results are stored, check residuals
		if hasattr(self.analysis, 'fit_results') and isinstance(self.analysis.fit_results, dict):
			for name, fit in self.analysis.fit_results.items():
				if name in self.analysis.processed:
					x, y = self.analysis.processed[name]
					residuals = y - fit
					# Overfitting: residuals too small
					if np.mean(np.abs(residuals)) < 1e-6:
						overfitting = True
						log_msgs.append(f"Overfitting detected in curve fitting for sample '{name}' (residuals too small)")
					# Underfitting: residuals too large
					if np.mean(np.abs(residuals)) > np.std(y):
						underfitting = True
						log_msgs.append(f"Underfitting detected in curve fitting for sample '{name}' (residuals too large)")

		# --- Actual ML: PCA/PLS checks ---
		# PCA explained variance ratio
		if hasattr(self.analysis, 'pca') and hasattr(self.analysis.pca, 'explained_variance_ratio_'):
			evr = self.analysis.pca.explained_variance_ratio_
			if np.any(evr > 0.99):
				overfitting = True
				log_msgs.append("Overfitting detected in PCA (explained variance ratio > 0.99)")
			if np.any(evr < 0.01):
				underfitting = True
				log_msgs.append("Underfitting detected in PCA (explained variance ratio < 0.01)")
		# PLS score
		if hasattr(self.analysis, 'pls') and hasattr(self.analysis.pls, 'score_'):
			score = self.analysis.pls.score_
			if score > 0.99:
				overfitting = True
				log_msgs.append("Overfitting detected in PLSRegression (score > 0.99)")
			if score < 0.1:
				underfitting = True
				log_msgs.append("Underfitting detected in PLSRegression (score < 0.1)")

		# --- Post-ML: Similarity metrics and p-values ---
		if hasattr(self.analysis, '_cached_results') and isinstance(self.analysis._cached_results, dict):
			results = self.analysis._cached_results
			sim_scores = [r.get('similarity_percent', 0) for r in results.values()]
			pvals = [r.get('p_value', 1) for r in results.values()]
			# Overfitting: similarity scores too high, p-values too low
			if len(sim_scores) > 0 and np.mean(sim_scores) > 99:
				overfitting = True
				log_msgs.append("Overfitting detected in similarity metrics (mean similarity > 99%)")
			if len(sim_scores) > 0 and np.mean(sim_scores) < 10:
				underfitting = True
				log_msgs.append("Underfitting detected in similarity metrics (mean similarity < 10%)")
			if len(pvals) > 0 and np.mean(pvals) < 0.001:
				overfitting = True
				log_msgs.append("Overfitting detected in permutation p-values (mean p-value < 0.001)")
			if len(pvals) > 0 and np.mean(pvals) > 0.5:
				underfitting = True
				log_msgs.append("Underfitting detected in permutation p-values (mean p-value > 0.5)")

		# --- DTW alignment checks ---
		# If DTW distances are stored, check for extremes
		if hasattr(self.analysis, 'dtw_distances') and isinstance(self.analysis.dtw_distances, dict):
			dtw_vals = list(self.analysis.dtw_distances.values())
			if len(dtw_vals) > 0 and np.mean(dtw_vals) < 1e-3:
				overfitting = True
				log_msgs.append("Overfitting detected in DTW alignment (mean DTW distance < 1e-3)")
			if len(dtw_vals) > 0 and np.mean(dtw_vals) > 1e3:
				underfitting = True
				log_msgs.append("Underfitting detected in DTW alignment (mean DTW distance > 1e3)")

		# --- General sanity checks ---
		# If all results are identical, we likely have overfitting or a bug
		if hasattr(self.analysis, '_cached_results') and isinstance(self.analysis._cached_results, dict):
			vals = [tuple(r.items()) for r in self.analysis._cached_results.values()]
			if len(vals) > 1 and all(v == vals[0] for v in vals):
				overfitting = True
				log_msgs.append("Overfitting detected: all results are identical across samples")

		# ---- Log all detected issues ----
		for msg in log_msgs:
			logging.warning(msg)
		if not log_msgs:
			logging.info("No overfitting or underfitting detected in any pipeline step.")

		return underfitting, overfitting
