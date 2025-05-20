# Basic Imports
import os
import logging
import json
import sys

# Data Science Imports
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from scipy.fft import fft, ifft
from scipy.optimize import curve_fit, dual_annealing
from scipy.stats import ttest_rel, pearsonr
from scipy.special import wofz
from skopt import gp_minimize
from skopt.space import Integer

# Machine Learning Imports
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# FastDTW
from fastdtw import fastdtw

# PyQt6 GUI Imports
from PyQt6.QtWidgets import (
	QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
	QWidget, QLabel, QPushButton, QFileDialog, 
	QMessageBox, QFrame, QProgressBar, QTableWidget, QTableWidgetItem, 
	QHeaderView, QCheckBox,
)
from PyQt6.QtGui import QFont, QIcon, QMovie, QPixmap
from PyQt6.QtCore import QTimer, QSize, Qt, QElapsedTimer

# ReportLab Imports (for PDF generation from results)
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# set up utils dir
utilsdir = os.path.expanduser("/Applications/Utilities")
utilsfolder = os.path.join(utilsdir, "LNVSI Tool Utilities")
if not os.path.exists(utilsfolder):
	os.makedirs(utilsfolder)

# Set up logging
log_dir = os.path.join(utilsfolder, "Logs")
LOG_FILE = os.path.join(log_dir, "LNVSI Tool.log")
if not os.path.exists(log_dir):
	os.makedirs(log_dir)
if not os.path.exists(LOG_FILE):
	with open(LOG_FILE, 'w') as f:
		pass

logging.basicConfig(
	filename=LOG_FILE,
	level=logging.DEBUG,
	format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d'
)

#Set up autosave
autosave_dir = os.path.join(utilsfolder, "Autosaved Data Entry Results")
if not os.path.exists(autosave_dir):
	os.makedirs(autosave_dir)
AUTO_SAVE_FILE = os.path.join(autosave_dir, "autosave.json")

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class UI(QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("ConoWare Levensaler Neogastropod Venomic Similarity Indexing Tool (version: alpha-testing)")
		self.setGeometry(100, 100, 800, 600)
		self.setMinimumSize(800, 600)
	   
		# Asset paths using resource_path for PyInstaller compatibility
		self.icon_path = resource_path("usr/assets/icon.icns") # this is macos version other iterations of this app will use .png
		self.background_path = resource_path("usr/assets/background.png")
		self.gif_path = resource_path("usr/assets/countdown.gif")
		# Set window icon
		if os.path.exists(self.icon_path):
			self.setWindowIcon(QIcon(self.icon_path))
		else:
			logging.error(f"Icon file not found: {self.icon_path}")
			raise FileNotFoundError(f"Icon file not found: {self.icon_path}")

		# Set styles
		styles_path = resource_path("usr/assets/styles.qss")
		if os.path.exists(styles_path):
			with open(styles_path, "r") as f:
				self.styles = f.read()
		else:
			self.styles = ""
			logging.warning(f"Stylesheet file not found: {styles_path}")

		# Set background image
		try:
			if os.path.exists(self.background_path):
				self.background_label = QLabel(self)
				pixmap = QPixmap(self.background_path)
				self.background_label.setPixmap(pixmap)
				self.background_label.setScaledContents(True)
				self.background_label.setGeometry(0, 0, self.width(), self.height())
				self.background_label.lower()
				self.background_label.show()
			else:
				logging.error(f"Background image file not found: {self.background_path}")
				raise FileNotFoundError(f"Background image file not found: {self.background_path}")
			try:
				if hasattr(self, 'background_label'):
					self.background_label.setGeometry(0, 0, self.width(), self.height())
			except AttributeError as e:
				logging.error("Event Resize failed")
				raise RuntimeError("Event Resize failed")
		
		except (Exception, FileNotFoundError, RuntimeError) as e:
			logging.error(f"Error setting background: {e}")
		
		# Set central widget
		self.central_widget = QWidget(self)
		self.setCentralWidget(self.central_widget)  # Set the central widget
		self.main_layout = QVBoxLayout()
		self.main_layout.setContentsMargins(0, 0, 0, 0)
		self.central_widget.setLayout(self.main_layout)

		# Stylesheet for window and background
		self.setStyleSheet(self.styles)

		# Other basic initializations - create in correct order
		# First create ReturnHome since other classes need it
		self.rh = ReturnHome(self)
		# Then create the other classes
		self.ae = AnalysisEngine(self)
		self.ef = ExportFindings(self)
		
		# Initialize progress_bar as None until it's created
		self.progress_bar = None

	def render_ui(self):

		# Set welcome_container to be centered with margins (middle third of window)
		self.welcome_container = QWidget(self)
		self.welcome_container.setStyleSheet(self.styles)
		self.main_layout.addWidget(self.welcome_container)

		# Center the container both horizontally and vertically with nice spacing
		margin_top_bottom = 60
		margin_sides = int(self.width() * 1/3)
		container_width = self.width() - 2 * margin_sides
		container_height = self.height() - 2 * margin_top_bottom
		container_x = margin_sides
		container_y = margin_top_bottom
		self.welcome_container.setGeometry(
			container_x,
			container_y,
			container_width,
			container_height
		)
		self.welcome_layout = QVBoxLayout(self.welcome_container)
		self.welcome_container.setLayout(self.welcome_layout)

		# Main welcome label
		self.welcome_label = QLabel(
			'<span class="main-title">Welcome to the LNVSI Tool (version alpha-testing)!</span><br>'
			'<span class="subtitle">Created by Luke Levensaler, 2025</span>'
		)
		self.welcome_label.setObjectName("welcomeLabel")
		self.welcome_label.setTextFormat(Qt.TextFormat.RichText)
		self.welcome_layout.addWidget(
			self.welcome_label,
			alignment=
			Qt.AlignmentFlag.AlignHCenter # Center the label
			| Qt.AlignmentFlag.AlignTop # (at top of the container)
		)
		self.welcome_label.setMinimumHeight(100)
		self.welcome_label.setMinimumWidth(300)
		
			
		# Sub welcome label
		self.sub_welcome_label = QLabel(
			"Let's analyze your Levensaler Assay-derived venomic datasets with the"
			"Levensaler Neogastropod Venomic Similarity Index's (LNVSI) machine learning-powered" 
			"statistical algorithm! The results will tell you if they share similarities to your" 
			"conotoxin positive control. \n"
			"If your device can handle it (see the ConoWare Project's website or our GitHub page)" 
			"and your peptidomes are conopeptide-like," 
			"you should install ConoBot AI to help you identify your proteomes' unique "
			"primary structure cysteine frameworks!"
		)
		self.sub_welcome_label.setFont(QFont("Arial", 16, QFont.Weight.Normal))
		self.sub_welcome_label.setStyleSheet(self.styles)
		self.sub_welcome_label.setMinimumWidth(200)
		self.sub_welcome_label.setMinimumHeight(400)
		self.sub_welcome_label.setWordWrap(True)
		self.welcome_layout.addWidget(
			self.sub_welcome_label,
			alignment=
			Qt.AlignmentFlag.AlignHCenter # Center the sublabel
			| Qt.AlignmentFlag.AlignVCenter # (in middle of the container)
		)

		# Upload CSV Button
		self.upload_button = QPushButton("Upload Your CSV File Here")
		self.upload_button.setStyleSheet(self.styles)
		self.upload_button.setFixedSize(250, 60)
		self.upload_button_font = QFont("Verdana", 16, QFont.Weight.Bold)
		self.upload_button.setFont(self.upload_button_font)
		self.upload_button.clicked.connect(self.confirm_csv_upload)
		self.welcome_layout.addWidget(
			self.upload_button,
			alignment=
			Qt.AlignmentFlag.AlignHCenter # Center the button
			| Qt.AlignmentFlag.AlignBottom # (at bottom of the container)
			)
		# Set object names for QSS styling
		self.central_widget.setObjectName("mainCentralWidget")
		# Welcome container
		self.welcome_container.setObjectName("welcomeContainer")
		self.welcome_label.setObjectName("welcomeLabel")
		self.sub_welcome_label.setObjectName("subWelcomeLabel")
		self.upload_button.setObjectName("uploadButton")
	
	def confirm_csv_upload(self):
		try:
			reply = QMessageBox.question(
				self,
				"Confirm Upload",
				"Are you sure you want to upload a new CSV file? This will start a new analysis and discard any unsaved results.",
				QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
			)
			if reply == QMessageBox.StandardButton.Yes:
				self.upload_button.setEnabled(False)  # Disable upload button during latency
				self.ae.upload_csv()
			else:
				# User aborted, just close the dialog and do nothing
				return
		except Exception as e:
			logging.error(f"Error confirming CSV upload: {e}")
			QMessageBox.critical(self, "Error", f"Error confirming CSV upload: {e}")
			raise RuntimeError(f"Error confirming CSV upload: {e}")

	def launch_data_analysis_mini_screen(self):

		# Properly remove and delete the welcome_container and all its children
		try:
			try:
				for child in self.welcome_container.findChildren(QWidget):
					child.setParent(None)
					child.deleteLater()
			except Exception as e:
				logging.error(f"Error clearing children of welcome_container: {e}")
				raise RuntimeError(f"Error clearing children of welcome_container: {e}")
			self.main_layout.removeWidget(self.welcome_container)
			self.welcome_container.setParent(None)
			self.welcome_container.deleteLater()
			del self.welcome_container
		except Exception as e:
			logging.error(f"Error with widget clearing: {e}")
			raise RuntimeError(f"Error removing welcome_container: {e}")
		
		# Create a new container for the data analysis with same dimensions as previous welcome_container
		margin_top_bottom = 60
		margin_sides = int(self.width() * 1/3)
		container_width = self.width() - 2 * margin_sides
		container_height = self.height() - 2 * margin_top_bottom
		container_x = margin_sides
		container_y = margin_top_bottom
		self.data_analysis_container = QWidget(self)
		self.data_analysis_container.setVisible(True)
		self.main_layout.addWidget(self.data_analysis_container)
		self.data_analysis_container.setGeometry(
			container_x,
			container_y,
			container_width,
			container_height
		)
		self.data_analysis_container.setStyleSheet(self.styles)
		self.data_analysis_container.setContentsMargins(0, 0, 0, 0)
		self.data_analysis_layout = QVBoxLayout(self.data_analysis_container)

		# Add a top label with breathing room and centered text
		self.data_analysis_label = QLabel("Running Similarity Engine...")
		self.data_analysis_label.setFont(QFont("Verdana", 26, QFont.Weight.ExtraBold))
		self.data_analysis_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
		self.data_analysis_label.setContentsMargins(0, 30, 0, 20)  # Top and bottom breathing room
		self.data_analysis_layout.addWidget(self.data_analysis_label)
		self.data_analysis_label.setStyleSheet(self.styles)

		# Add a horizontal line (QFrame) after the label for separation
		self.line = QFrame()
		self.line.setObjectName("line")
		self.line.setFrameShape(QFrame.Shape.HLine)
		self.line.setFrameShadow(QFrame.Shadow.Sunken)
		self.data_analysis_layout.addWidget(self.line)

		# Centered Perfectly Progress Bar Container
		self.progress_container = QWidget(self.data_analysis_container)
		self.progress_container.setFixedHeight(105)
		self.progress_container.setFixedWidth(400)
		self.progress_container.setContentsMargins(0, 0, 0, 0) # No margins for the progress bar
		self.data_analysis_layout.addStretch(1)
		self.data_analysis_layout.addWidget(
			self.progress_container, 
			alignment=
			Qt.AlignmentFlag.AlignHCenter
			| Qt.AlignmentFlag.AlignVCenter
		)
		self.data_analysis_layout.addStretch(1)
	   
		# Progress Bar Layout
		self.progress_layout = QHBoxLayout(self.progress_container)
		self.progress_container.setLayout(self.progress_layout)
		self.progress_layout.setContentsMargins(0, 0, 0, 0) # No margins for the progress bar

		# Add a label for the progress bar (top 1/2 of progress layout)
		# Ensure analysis_progress_bar is created before set_splash_text is called
		self.analysis_progress_bar = QProgressBar(self.progress_container)
		self.analysis_progress_bar.setRange(0, 100)
		self.analysis_progress_bar.setValue(0)
		self.analysis_progress_bar.setFixedHeight(55)
		self.analysis_progress_bar.setFixedWidth(300)
		self.analysis_progress_bar.setStyleSheet(self.styles)

		# Set progress_bar reference for compatibility
		self.progress_bar = self.analysis_progress_bar

		self.splash_text = self.set_splash_text() # splash text dependent on actual progress
		self.progress_label = QLabel(self.splash_text)
		self.progress_label.setFont(QFont("Helvetica", 16, QFont.Weight.Normal))
		self.progress_label.setFixedHeight(50)
		self.progress_label.setFixedWidth(300)
		self.progress_label.setStyleSheet(self.styles)
		self.progress_layout.addWidget(
			self.progress_label, 
			alignment=
			Qt.AlignmentFlag.AlignHCenter 
			| Qt.AlignmentFlag.AlignTop
		)

		# Add the progress bar after the label
		self.progress_layout.addWidget(
			self.analysis_progress_bar, 
			alignment=
			Qt.AlignmentFlag.AlignBottom
			| Qt.AlignmentFlag.AlignHCenter
		)
		# Add countdown gif to the right of the progress bar
		self.show_countdown_gif (
			self.progress_layout, 
			self.progress_container, 
			trigger_type='progress'
		)

		# Cancel button at bottom left of entire container
		self.cancel_button = QPushButton("Cancel")
		self.cancel_button.setFont(QFont("Verdana", 16, QFont.Weight.Bold))
		self.cancel_button.setStyleSheet(self.styles)
		self.cancel_button.setContentsMargins(0, 30, 0, 20)  # top and left breathing room
		self.cancel_button.setFixedSize(250, 60)
		self.data_analysis_layout.addWidget(
			self.cancel_button,
			alignment=
			Qt.AlignmentFlag.AlignLeft # Left align the button
			| Qt.AlignmentFlag.AlignBottom # (at bottom of the container)
		)
		
		self.cancel_button.clicked.connect(self.confirm_analysis_cancellation)
		# Data analysis container
		self.data_analysis_container.setObjectName("dataAnalysis")
		self.data_analysis_label.setObjectName("analysisLabel")
		self.progress_container.setObjectName("progressContainer")
		self.progress_label.setObjectName("progressLabel")
		self.analysis_progress_bar.setObjectName("progressBar")
		self.cancel_button.setObjectName("cancelButton")

	def set_splash_text(self):
		# Set the splash text based on the progress
		progress = self.analysis_progress_bar.value()
		if progress == 0:
			return "Initializing..."
		elif 0 < progress < 10:
			return "Preparing Data..."
		elif 10 <= progress < 20:
			return "Reading CSV File..."
		elif 20 <= progress < 25:
			return "Preprocessing Curves..."
		elif 25 <= progress < 30:
			return "Applying Lowpass Filter..."
		elif 30 <= progress < 35:
			return "Smoothing Data..."
		elif 35 <= progress < 40:
			return "Curve Smoothing Complete."
		elif 40 <= progress < 50:
			return "Segmenting Spectral Regions..."
		elif 50 <= progress < 60:
			return "Deconvolving Voigt Profiles..."
		elif 60 <= progress < 65:
			return "Aligning Curves (DTW)..."
		elif 65 <= progress < 70:
			return "Computing Similarity Metrics..."
		elif 70 <= progress < 75:
			return "Calculating AUC and Statistics..."
		elif 75 <= progress < 80:
			return "Running Statistical Tests..."
		elif 80 <= progress < 90:
			return "Analyzing All Samples..."
		elif 90 <= progress < 95:
			return "Saving Results..."
		elif 95 <= progress < 100:
			return "Finalizing Analysis..."
		elif progress == 100:
			# Adaptive latency: ensure at least 3 seconds 
			# (but allow more if system is busy)
			QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
			if hasattr(self, 'cancel_button'):
				self.cancel_button.setEnabled(False)  # Disable cancel button during latency
			self._finalize_timer = QElapsedTimer()
			self._finalize_timer.start()
			def try_finalize_results():
				# Process pending events to keep UI smooth
				QApplication.processEvents()
				if self._finalize_timer.elapsed() >= 3000:
					QApplication.restoreOverrideCursor()
					self.results()
				else:
					QTimer.singleShot(100, try_finalize_results)
			QTimer.singleShot(100, try_finalize_results)
			return "Finalizing Results..."
		return "Idle."

	def results (self):

		# Properly remove and delete the data analysis container and all its children
		try:
			for child in self.data_analysis_container.findChildren(QWidget):
				child.setParent(None)
				child.deleteLater()
			self.main_layout.removeWidget(self.data_analysis_container)
			self.data_analysis_container.setParent(None)
			self.data_analysis_container.deleteLater()
			del self.data_analysis_container
		except Exception as e:
			logging.error(f"Error removing data_analysis_container: {e}")
			raise RuntimeError(f"Error removing data_analysis_container: {e}")
		
		# Create a new container for the results with same dimensions as previous two containers
		self.results_container = QWidget(self)
		self.main_layout.addWidget(self.results_container)
		margin_top_bottom = 60
		margin_sides = int(self.width() * 1/3)
		container_width = self.width() - 2 * margin_sides
		container_height = self.height() - 2 * margin_top_bottom
		container_x = margin_sides
		container_y = margin_top_bottom
		self.results_container.setGeometry(
			container_x,
			container_y,
			container_width,
			container_height
		)
		self.results_container.setVisible(True)
		self.results_container.setStyleSheet(self.styles)
		self.results_container.setContentsMargins(0, 0, 0, 0)
		self.results_layout = QVBoxLayout(self.results_container)

		# Add a top label for the results screen, centered with breathing room and QSS dash style
		self.results_label = QLabel("Algorithm Output - Results")
		self.results_label.setFont(QFont("Helvetica", 16, QFont.Weight.Normal))
		self.results_label.setFixedHeight(50)
		self.results_label.setFixedWidth(300)
		self.results_label.setStyleSheet(self.styles)
		self.results_layout.addWidget(
			self.results_label,
			alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
		)

		# Actual results table portal where results will be displayed for each analyte 
		# (relative to positive control)
		self.results_portal = QWidget(self.results_container)
		self.results_portal.setGeometry(500, 500, 500, 500)
		self.results_portal.setStyleSheet(self.styles)
		self.results_portal.setContentsMargins(0, 50, 50, 0)
		self.results_layout.addWidget(self.results_label)
		self.results_portal_layout = QVBoxLayout(self.results_portal)
		self.results_portal.setLayout(self.results_portal_layout)

		# Actual results table
		self.results_table = QTableWidget(self.results_portal)
		self.results_table.setFixedSize(500, 500)  # Set a fixed size for the table
		self.results_portal_layout.addWidget(self.results_table)
		self.results_table.setStyleSheet(self.styles)
		self.results_table.setContentsMargins(0, 0, 0, 0)
		self.results_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
		self.results_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
		self.results_table.setColumnCount(4)
		self.results_table.setRowCount(self.ae.num_analytes)
		self.results_table.setHorizontalHeaderLabels(["Sample", "Conotoxin Continuity", "Similarity %", "p-value"])
		
		# Use Arial Black for all text in the table
		arial_black = QFont("Arial Black", 10)
		arial_black_bold = QFont("Arial Black", 10, QFont.Weight.Bold)
		self.results_table.horizontalHeader().setFont(arial_black_bold)
		self.results_table.verticalHeader().setFont(arial_black_bold)
	   
		# Center align headers text
		self.results_table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
		self.results_table.verticalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
		
		# Auto-resize first column to fit content, but only within the visible frame
		self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
		for col in range(1, 4):
			self.results_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
			self.results_table.setColumnWidth(col, 120)
		self.results_table.setColumnWidth(0, 200)
		self.results_table.horizontalHeader().setSectionResizeMode(Qt.Orientation.Horizontal, QHeaderView.ResizeMode.Fixed)
		self.results_table.verticalHeader().setSectionResizeMode(Qt.Orientation.Vertical, QHeaderView.ResizeMode.Fixed)
		self.results_table.horizontalHeader().setSectionsMovable(False)
		self.results_table.verticalHeader().setSectionsMovable(False)
		self.results_table.horizontalHeader().setStretchLastSection(True)
		self.results_table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
		self.results_table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
		self.results_table.setCornerButtonEnabled(False)
		self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
		self.results_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
		self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
		self.results_table.setShowGrid(True)
		self.results_table.setWordWrap(True)
		self.results_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
		self.results_table.horizontalHeader().setVisible(True)
		self.results_table.verticalHeader().setVisible(True)

		# Freeze first column and header row
		self.results_table.horizontalHeader().setSectionsClickable(False)
		self.results_table.verticalHeader().setSectionsClickable(False)

		# Populate the table with results for each analyte (excluding positive control)
		if hasattr(self.ae, 'sample_names') and hasattr(self.ae, 'analyze_all_samples'):
			# Run analysis and collect results
			results_dict = self.ae.analyze_all_samples()
			# Fill the table
			for row, sample in enumerate(self.ae.sample_names):
				result = results_dict.get(sample, {})
				for col, value in enumerate([
					str(sample),
					"Yes" if result.get("conotoxin_like") else "No",
					f"{result.get('similarity_percent', 0):.2f}",
					f"{result.get('p_value', 1):.3g}"
				]):
					item = QTableWidgetItem(value)
					# Ensure all cell text is center-aligned
					item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
					item.setFont(arial_black if col > 0 else arial_black_bold)
					self.results_table.setItem(row, col, item)

			# Set default alignment for all cells (including empty ones)
			for row in range(self.results_table.rowCount()):
				for col in range(self.results_table.columnCount()):
					item = self.results_table.item(row, col)
					if item is None:
						item = QTableWidgetItem("")
						item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
						self.results_table.setItem(row, col, item)
					else:
						item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

		# Download button
		self.download_button = QPushButton("Select Method of Exporting Results")
		self.download_button.setFont(QFont("Verdana", 16, QFont.Weight.Bold))
		self.download_button.setStyleSheet(self.styles)
		self.download_button.setContentsMargins(0, 30, 20, 0) #top and right breathing room
		self.download_button.setFixedSize(250, 60) 
		self.results_layout.addWidget(
			self.download_button,
			alignment=
			Qt.AlignmentFlag.AlignRight # Right align the button
			| Qt.AlignmentFlag.AlignBottom # (at bottom of the container)
		)
		self.download_button.clicked.connect(self.confirm_results_download)  # Add countdown gif to the right of the download button
		self.show_countdown_gif(
			self.results_layout, 
			self.results_container, 
			trigger_widget=self.download_button, 
			trigger_type='download'
		)

		# Discard button
		self.discard_button = QPushButton("Discard Results")
		self.discard_button.setFont(QFont("Verdana", 16, QFont.Weight.Bold))
		self.discard_button.setStyleSheet(self.styles)
		self.discard_button.setContentsMargins(20, 30, 0, 0)  # top and left breathing room
		self.discard_button.setFixedSize(250, 60)
		self.results_layout.addWidget(
			self.discard_button,
			alignment=
			Qt.AlignmentFlag.AlignLeft # Left align the button
			| Qt.AlignmentFlag.AlignBottom # (at bottom of the container)
		)
		self.discard_button.clicked.connect(self.confirm_results_discard)  # Add countdown gif to the right of the download button
		self.show_countdown_gif(
			self.results_layout, 
			self.results_container, 
			trigger_widget=self.discard_button, 
			trigger_type='discard'
		)
		# Results container
		self.results_container.setObjectName("resultsContainer")
		self.results_label.setObjectName("resultLabel")
		self.results_portal.setObjectName("resultsWidget")
		self.results_table.setObjectName("resultsTable")
		self.download_button.setObjectName("downloadButton")
		self.discard_button.setObjectName("discardButton")

	# Connect the download button to the export logic, but confirm with user first
	def confirm_results_download(self):
		try:
			reply = QMessageBox.question(
				self,
				"Confirm Download",
				"Are you sure you want to download your finalized LNVSI results?",
				QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
			)
			if reply == QMessageBox.StandardButton.Yes:
				self.download_button.setEnabled(False)  # Disable download button during latency
				self.ef.downloader_ui()
			else:
				# User aborted, just close the dialog and do nothing
				return
		except Exception as e:
			logging.error(f"Error confirming download: {e}")
			QMessageBox.critical(self.ui, "Error", f"Error confirming download: {e}")
			raise RuntimeError(f"Error confirming download: {e}")

	def confirm_results_discard(self):
		try:
			reply = QMessageBox.question(
				self,
				"Confirm Discard",
				"Are you sure you want to irrecoverably discard your finalized LNVSI results?",
				QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
			)
			if reply == QMessageBox.StandardButton.Yes:
				self.discard_button.setEnabled(False)  # Disable discard button during latency
				self.rh.return_home_from_discard()
			else:
				# User aborted, just close the dialog and do nothing
				return
		except Exception as e:
			logging.error(f"Error confirming discard: {e}")
			QMessageBox.critical(self.ui, "Error", f"Error confirming discard: {e}")
			raise RuntimeError(f"Error confirming discard: {e}")

	def confirm_analysis_cancellation(self):
		try:
			reply = QMessageBox.question(
				self,
				"Confirm Discard",
				"Are you sure you want to irrecoverably cancel your LNVSI analysis?",
				QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
			)
			if reply == QMessageBox.StandardButton.Yes:
				self.cancel_button.setEnabled(False)  # Disable cancel button during latency
				self.rh.return_home_from_cancel()
			else:
				# User aborted, just close the dialog and do nothing
				return
		except Exception as e:
			logging.error(f"Error confirming cancel: {e}")
			QMessageBox.critical(self, "Error", f"Error confirming cancel: {e}")
			raise RuntimeError(f"Error confirming cancel: {e}")
	
	def show_countdown_gif(self, parent_layout, parent_container, trigger_widget=None, trigger_type='progress'):
		"""
		Adds a countdown gif container to the given parent_layout (QLayout) and parent_container (QWidget).
		The gif is always square, fills its container, and is placed at the right and vertically centered
		relative to the parent_layout. Optionally, can be triggered by a widget (e.g., download button).
		"""
		gif_size = 55

		# Create gif container and label
		gif_container = QWidget(parent_container)
		gif_container.setFixedSize(gif_size, gif_size)
		gif_container.setStyleSheet(self.styles)
		gif_label = QLabel(gif_container)
		gif_label.setFixedSize(gif_size, gif_size)
		gif_label.setStyleSheet(self.styles)
		gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

		# Load gif if available
		if os.path.exists(self.gif_path):
			movie = QMovie(self.gif_path)
			movie.setScaledSize(QSize(gif_size, gif_size))
			gif_label.setMovie(movie)
			movie.jumpToFrame(0)
			movie.start()
		else:
			logging.error(f"Countdown gif not found: {self.gif_path}")

		# Add to layout at the right and vcenter
		if isinstance(parent_layout, QHBoxLayout):
			parent_layout.addWidget(gif_container, alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
		elif isinstance(parent_layout, QVBoxLayout):
			# If trigger_widget is given, place gif right of it, else just at bottom right
			if trigger_widget is not None:
				# Insert after the trigger_widget
				index = parent_layout.indexOf(trigger_widget)
				if index != -1:
					parent_layout.insertWidget(index + 1, gif_container, alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
				else:
					parent_layout.addWidget(gif_container, alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
			else:
				parent_layout.addWidget(gif_container, alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
		else:
			logging.error("Parent layout is not a QHBoxLayout or QVBoxLayout.")


class AnalysisEngine:

    def __init__(self, ui):
        self.filename = None
        self.ui = ui  # Use the passed UI instance directly
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
            file_name, _ = QFileDialog.getOpenFileName(
                self.ui, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options
            )
            logging.info("File dialog opened for CSV selection.")
            if file_name:
                self.filename = file_name
                logging.info(f"CSV file selected: {self.filename}")
               
			    # Show success message (auto-close after 5 seconds)
                msg = QMessageBox(self.ui)
                self._upload_msgbox = msg 
                msg.setWindowTitle("Upload Successful")
                msg.setText("CSV file uploaded successfully. Initializing analysis...")
                msg.setIcon(QMessageBox.Icon.Information)
                msg.setStandardButtons(QMessageBox.StandardButton.NoButton)
                msg.show()
                logging.info("Upload success message box shown.")
                QApplication.processEvents()
                QTimer.singleShot(5000, lambda: msg.close())
                logging.info("Upload success message box will auto-close in 5 seconds.")
                
				# Launch analysis UI immediately
                self.ui.launch_data_analysis_mini_screen()
                logging.info("Data analysis mini screen launched.")
                self.ui.progress_bar = self.ui.analysis_progress_bar
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                self._upload_timer = QElapsedTimer()
                self._upload_timer.start()
                logging.info("Upload timer started for artificial latency.")
                def finish_upload():
                    QApplication.processEvents()
                    if self._upload_timer.elapsed() >= 3000:
                        QApplication.restoreOverrideCursor()
                        logging.info("Upload latency complete, proceeding to read_csv.")
                        self.read_csv()
                    else:
                        QTimer.singleShot(100, finish_upload)
                QTimer.singleShot(100, finish_upload)
                logging.info("CSV file upload completed, waiting for latency to finish.")
                self.ui.progress_bar.setValue(0)
                QApplication.processEvents()

        except Exception as e:
            logging.error(f"Error uploading CSV file: {e}")
            QMessageBox.critical(self.ui, "Error", f"Error uploading CSV file: {e}")
            raise RuntimeError(f"Error uploading CSV file: {e}")

    def read_csv(self):
        try:
            logging.info("Starting read_csv method.")
            QApplication.processEvents()
            df = pd.read_csv(self.filename)
            logging.info("CSV file loaded into DataFrame.")
            columns_lower = [col.lower() for col in df.columns]
            nm_idx = next((i for i, col in enumerate(columns_lower) if col == 'nm'), 0)
            pos_ctrl_idx = next((i for i, col in enumerate(columns_lower) if col == 'positive control'), 1)
            if nm_idx != 0 or pos_ctrl_idx != 1:
                cols = list(df.columns)
                nm_col = cols[nm_idx]
                pos_ctrl_col = cols[pos_ctrl_idx]
                analyte_cols = [c for i, c in enumerate(cols) if i not in (nm_idx, pos_ctrl_idx)]
                new_order = [nm_col, pos_ctrl_col] + analyte_cols
                df = df[new_order]
                logging.info("Columns reordered to match expected format.")
            wavelengths = df.iloc[:, 0].values
            samples = list(df.columns[1:])
            data = {name: df[name].values for name in samples}
            self.wavelengths = wavelengths
            self.absorbance = data
            logging.info(f"Wavelengths and absorbance data extracted. {len(samples)} samples found.")
            if len(samples) < 2:
                logging.error("CSV must contain at least 'positive control' and one analyte column after 'nm'.")
                raise ValueError("CSV must contain at least 'positive control' and one analyte column after 'nm'.")
            self.positive_control = samples[0]
            self.sample_names = samples[1:]
            self.num_analytes = len(self.sample_names)
            logging.info(f"Positive control: {self.positive_control}. Analytes: {self.sample_names}")
            self.processed = self.preprocess_all_curves(wavelengths, data)
            self.ui.progress_bar.setValue(10)
            QApplication.processEvents()
            logging.info(f"CSV file read successfully: {self.filename} | {self.num_analytes} analytes found. | Beginning smoothing..")
        except Exception as e:
            logging.error(f"Error in read_csv: {e}")
            QMessageBox.critical(self.ui, "Error Reading CSV", f"An error occurred while reading your CSV file. Please check the format and try again.\n\nError: {e}")
            self.rh.return_home_from_error()
            return

    def preprocess_all_curves(self, wavelengths, data):
        try:
            logging.info("Starting preprocess_all_curves method.")
            QApplication.processEvents()
            processed = {}
            for name, y in data.items():
                logging.info(f"Smoothing curve for sample: {name}")
                QApplication.processEvents()
                x_new, y_final = self.smooth_curve(wavelengths, y)
                processed[name] = (x_new, y_final)
                logging.info(f"Curve for {name} smoothed and stored.")
            self.ui.progress_bar.setValue(20)
            QApplication.processEvents()
            logging.info("All curves smoothed. Exiting preprocess_all_curves.")
            return processed
        except Exception as e:
            logging.error(f"Error in preprocess_all_curves: {e}")
            QMessageBox.critical(self.ui, "Error Preprocessing Data", f"An error occurred while preprocessing your data.\n\nError: {e}")
            self.rh.return_home_from_error()
            return {}

    def smooth_curve(self, x, y):
        try:
            logging.info("Starting smooth_curve method.")
            QApplication.processEvents()
            def lowpass_filter(y, cutoff_ratio=0.1):
                logging.info("Applying lowpass filter.")
                N = len(y)
                yf = fft(y)
                cutoff = int(N * cutoff_ratio)
                yf[cutoff:-cutoff] = 0
                self.ui.progress_bar.setValue(25)
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
            result = gp_minimize(sg_error, search_space, n_calls=20, random_state=0)
            win_opt = int(result.x[0])
            if win_opt % 2 == 0:
                win_opt += 1
            poly_opt = int(result.x[1])
            poly_opt = max(2, min(poly_opt, win_opt-1))
            self.ui.progress_bar.setValue(30)
            QApplication.processEvents()
            logging.info(f"Optimal Savitzky-Golay params: window={win_opt}, poly={poly_opt}")
            y_final = savgol_filter(y_lp, window_length=win_opt, polyorder=poly_opt)
            self.ui.progress_bar.setValue(35)
            QApplication.processEvents()
            logging.info("Savitzky-Golay smoothing complete.")
            return x_new, y_final
        except Exception as e:
            logging.error(f"Error in smooth_curve: {e}")
            QMessageBox.critical(self.ui, "Error Smoothing Curve", f"An error occurred while smoothing a curve.\n\nError: {e}")
            self.rh.return_home_from_error()
            return x, y

    def segment_regions(self, x, y):
        try:
            logging.info("Starting segment_regions method.")
            QApplication.processEvents()
            regions = {
                'NUV': (350, 450),
                'VIS': (450, 750),
                'NIR': (750, 1020)
            }
            def segment(x, y, start, end):
                mask = (x >= start) & (x <= end)
                return x[mask], y[mask]
            self.ui.progress_bar.setValue(40)
            QApplication.processEvents()
            logging.info("Segmenting regions: NUV, VIS, NIR.")
            segmented = {
                region: segment(x, y, *bounds)
                for region, bounds in regions.items()
            }
            logging.info("Regions segmented successfully.")
            return segmented
        except Exception as e:
            logging.error(f"Error in segment_regions: {e}")
            QMessageBox.critical(self.ui, "Error Segmenting Regions", f"An error occurred while segmenting the spectrum.\n\nError: {e}")
            self.rh.return_home_from_error()
            return {}

    def deconvolve_voigt(self, x, y):
        try:
            logging.info("Starting deconvolve_voigt method.")
            QApplication.processEvents()
            def voigt(x, amp, cen, sigma, gamma):
                z = ((x - cen) + 1j*gamma) / (sigma * np.sqrt(2))
                return amp * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))
            p0 = [y.max(), x[np.argmax(y)], 10, 1]
            logging.info(f"Initial Voigt fit parameters: {p0}")
            popt, _ = curve_fit(voigt, x, y, p0=p0, maxfev=10000)
            fit = voigt(x, *popt)
            self.ui.progress_bar.setValue(50)
            QApplication.processEvents()
            logging.info(f"Voigt fit complete. Parameters: {popt}")
            return fit
        except Exception as e:
            logging.error(f"Error in deconvolve_voigt: {e}")
            QMessageBox.critical(self.ui, "Error Deconvolving Voigt Profile", "An error occurred while fitting a Voigt profile. This may be due to challenging data or convergence issues. Please check your input data and try again.\n\nError: {}".format(e))
            self.rh.return_home_from_error()
            return y

    def align_curves_dtw(self, ref, query):
        try:
            logging.info("Starting align_curves_dtw method.")
            QApplication.processEvents()
            distance, path = fastdtw(ref, query)
            logging.info(f"DTW alignment complete. Distance: {distance}")
            aligned = np.interp(np.arange(len(ref)), [p[1] for p in path], [query[p[1]] for p in path])
            self.ui.progress_bar.setValue(60)
            QApplication.processEvents()
            logging.info("Curves aligned using DTW.")
            return aligned
        except Exception as e:
            logging.error(f"Error in align_curves_dtw: {e}")
            QMessageBox.critical(self.ui, "Error Aligning Curves", f"An error occurred while aligning curves using DTW.\n\nError: {e}")
            self.rh.return_home_from_error()
            return query

    def compute_similarity_metrics(self, y_ctrl, y_sample):
        try:
            logging.info("Starting compute_similarity_metrics method.")
            QApplication.processEvents()
            scaler = StandardScaler()
            X = np.vstack([y_ctrl, y_sample])
            X_scaled = scaler.fit_transform(X)
            logging.info("Data standardized for similarity metrics.")
            pca = PCA(n_components=1)
            pca.fit(X_scaled)
            pca_score = np.abs(pca.components_[0][0] - pca.components_[0][1])
            self.ui.progress_bar.setValue(65)
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
            self.ui.progress_bar.setValue(70)
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
            self.ui.progress_bar.setValue(75)
            QApplication.processEvents()
            logging.info(f"Similarity score: {sim_score}")
            try:
                t_stat, p_val = ttest_rel(y_ctrl, y_sample)
                logging.info(f"t-test p-value: {p_val}")
            except Exception:
                p_val = 1.0
                logging.info("t-test failed, p-value set to 1.0.")
            return sim_score, p_val
        except Exception as e:
            logging.error(f"Error in compute_similarity_metrics: {e}")
            QMessageBox.critical(self.ui, "Error Computing Similarity", f"An error occurred while computing similarity metrics.\n\nError: {e}")
            self.rh.return_home_from_error()
            return 0.0, 1.0

    def analyze_all_samples(self):
        try:
            logging.info("Starting analyze_all_samples method.")
            QApplication.processEvents()
            results = {}
            ctrl_name = self.positive_control
            logging.info(f"Using positive control: {ctrl_name}")
            x_ctrl, y_ctrl = self.processed[ctrl_name]
            regions_ctrl = self.segment_regions(x_ctrl, y_ctrl)
            logging.info("Segmented positive control regions.")
            y_nuv_dec = self.deconvolve_voigt(*regions_ctrl['NUV'])
            logging.info("Deconvolved NUV region for control.")
            y_nir_dec = self.deconvolve_voigt(*regions_ctrl['NIR'])
            logging.info("Deconvolved NIR region for control.")
            x_vis, y_vis = regions_ctrl['VIS']
            y_ctrl_full = np.concatenate([y_nuv_dec, y_vis, y_nir_dec])
            logging.info("Constructed full control curve.")
            self.ui.progress_bar.setValue(80)
            QApplication.processEvents()
            for name in self.sample_names:
                logging.info(f"Analyzing sample: {name}")
                QApplication.processEvents()
                x_s, y_s = self.processed[name]
                regions_s = self.segment_regions(x_s, y_s)
                logging.info(f"Segmented regions for sample: {name}")
                y_nuv_s_dec = self.deconvolve_voigt(*regions_s['NUV'])
                logging.info(f"Deconvolved NUV region for sample: {name}")
                y_nir_s_dec = self.deconvolve_voigt(*regions_s['NIR'])
                logging.info(f"Deconvolved NIR region for sample: {name}")
                x_vis_s, y_vis_s = regions_s['VIS']
                y_full = np.concatenate([y_nuv_s_dec, y_vis_s, y_nir_s_dec])
                logging.info(f"Constructed full curve for sample: {name}")
                try:
                    y_aligned = self.align_curves_dtw(y_ctrl_full, y_full)
                    logging.info(f"Aligned sample {name} to control.")
                except Exception:
                    y_aligned = y_full
                    logging.info(f"Alignment failed for {name}, using unaligned curve.")
                sim_score, p_val = self.compute_similarity_metrics(y_ctrl_full, y_aligned)
                logging.info(f"Similarity metrics for {name}: score={sim_score}, p={p_val}")
                result = {
                    "analyte": name,
                    "conotoxin_like": bool(p_val < 0.05),
                    "similarity_percent": sim_score,
                    "p_value": p_val
                }
                results[name] = result
            self.ui.progress_bar.setValue(90)
            QApplication.processEvents()
            logging.info("All samples analyzed. Preparing to save results.")
            if results and all(isinstance(v, dict) and v for v in results.values()):
                all_results = []
                if os.path.exists(AUTO_SAVE_FILE):
                    try:
                        with open(AUTO_SAVE_FILE, "r") as f:
                            prev = f.read().strip()
                            if prev:
                                entries = prev.split("\n\n---\n\n")
                                for entry in entries:
                                    if entry.strip():
                                        all_results.append(json.loads(entry))
                        logging.info("Loaded previous autosave entries.")
                    except Exception as e:
                        logging.warning(f"Could not load previous autosave entries: {e}")
                all_results.insert(0, results)
                with open(AUTO_SAVE_FILE, "w") as f:
                    for i, entry in enumerate(all_results):
                        f.write(json.dumps(entry, indent=2))
                        if i < len(all_results) - 1:
                            f.write("\n\n---\n\n")
                self.ui.progress_bar.setValue(95)
                QApplication.processEvents()
                logging.info(f"Analysis complete for {self.num_analytes} analytes. Results saved.")
            else:
                logging.warning("No valid results to save. Autosave skipped.")
            self.ui.progress_bar.setValue(100)
            QApplication.processEvents()
            logging.info("analyze_all_samples method complete.")
            return results
        except Exception as e:
            logging.error(f"Error in analyze_all_samples: {e}")
            QMessageBox.critical(self.ui, "Error During Analysis", f"An error occurred during the analysis.\n\nError: {e}")
            self.rh.return_home_from_error()
            return {}

class ExportFindings:
	def __init__(self, ui=None):
		self.ui = ui  # Use the passed UI instance directly
		# self.rh will be assigned when needed from the existing instance

	def downloader_ui(self):
		# Properly remove and delete the results_portal and all its children
		try:
			for child in self.ui.results_portal.findChildren(QWidget):
				child.setParent(None)
				child.deleteLater()
			self.ui.results_layout.removeWidget(self.ui.results_portal)
			self.ui.results_portal.setParent(None)
			self.ui.results_portal.deleteLater()
			del self.ui.results_portal
		except Exception as e:
			logging.error(f"Error removing results_portal: {e}")
			raise RuntimeError(f"Error removing results_portal: {e}")

		# Create a new results_portal widget with same dimensions and margins
		margin_top_bottom = 60
		margin_sides = int(self.ui.width() * 1/3)
		container_width = self.ui.width() - 2 * margin_sides
		container_height = self.ui.height() - 2 * margin_top_bottom
		container_x = margin_sides
		container_y = margin_top_bottom

		self.download_container = QWidget(self.ui)
		self.download_container.setGeometry(
			container_x,
			container_y,
			container_width,
			container_height
		)
		self.download_container.setStyleSheet(self.ui.styles)
		self.download_container.setContentsMargins(0, 50, 50, 0)
		self.download_layout = QVBoxLayout(self.download_container)
		self.download_container.setLayout(self.download_layout)
		self.ui.results_layout.addWidget(self.download_container)

		# Add a QLabel with a dotted line style as before
		self.download_label = QLabel("Save results as...")
		self.download_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
		self.download_label.setStyleSheet(self.ui.styles)
		self.download_layout.addWidget(self.download_label)
		self.download_label.setContentsMargins(0, 30, 0, 20)
		self.download_label.setFont(QFont("Helvetica", 16, QFont.Weight.Bold))
		self.download_label.setFixedHeight(50)
		self.download_label.setFixedWidth(300)
		self.download_label.setWordWrap(True)

		# Add a horizontal line (QFrame) after the label for separation
		self.line = QFrame()
		self.line.setObjectName("line")
		self.line.setFrameShape(QFrame.Shape.HLine)
		self.line.setFrameShadow(QFrame.Shadow.Sunken)
		self.download_layout.addWidget(self.line)
		self.download_layout.setContentsMargins(0, 0, 0, 0)
		self.download_layout.setSpacing(0)
		self.download_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
		self.download_layout.addStretch(1)

		# Download Options - Multiple Selection (QCheckBox list)
		self.download_options_container = QWidget(self.download_container)
		self.download_layout.addWidget(
			self.download_options_container,
			alignment=
			Qt.AlignmentFlag.AlignHCenter
			| Qt.AlignmentFlag.AlignVCenter
		)
		self.download_options_container.setStyleSheet(self.ui.styles)
		self.download_options_container.setContentsMargins(0, 0, 0, 0)
		self.download_options_container.setFixedHeight(400)
		self.download_options_container.setFixedWidth(200)

		# Add a final download button
		self.final_download_button = QPushButton("Download File(s)", self.download_container)
		self.final_download_button.setFont(QFont("Verdana", 16, QFont.Weight.Bold))
		self.final_download_button.setStyleSheet(self.ui.styles)
		self.final_download_button.setFixedSize(250, 60)
		self.download_layout.addWidget(
			self.final_download_button,
			alignment=
			Qt.AlignmentFlag.AlignHCenter
			| Qt.AlignmentFlag.AlignBottom
		)
		self.final_download_button.clicked.connect(self.download_results)

		# Checkbox layout
		self.download_options_layout = QVBoxLayout(self.download_options_container)
		self.download_options_container.setLayout(self.download_options_layout)
		self.download_options_layout.setContentsMargins(0, 0, 0, 0)
		self.download_options_layout.setSpacing(5)

		# Actual checkbox list
		try:
			self.possibilities = [
				"CSV",
				"PDF",
				"XLSX (Apple Numbers or Microsoft Excel)"
			]
			self.option_checkboxes = []  # Reset in case of re-entry
			for opt in self.possibilities:
				cb = QCheckBox(opt, self.download_options_container)
				cb.setStyleSheet(self.ui.styles)
				cb.setFont(QFont("Verdana", 14, QFont.Weight.Normal))
				cb.setContentsMargins(40, 50, 50, 0)
				cb.setFixedHeight(50)
				cb.setFixedWidth(300)
				cb.setChecked(False)
				cb.setCheckable(True)
				self.option_checkboxes.append(cb)
				# Alignment pattern as before
				if opt == "CSV":
					self.download_options_layout.addWidget(
						cb,
						alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
					)
				elif opt == "PDF":
					self.download_options_layout.addWidget(
						cb,
						alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
					)
				elif opt == "XLSX (Apple Numbers or Microsoft Excel)":
					self.download_options_layout.addWidget(
						cb,
						alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft
					)
		except Exception as e:
			logging.error(f"Error creating checklist: {e}")
			QMessageBox.critical(self.ui, "Error", f"Error creating checklist: {e}")
			raise RuntimeError(f"Error creating checkbox options: {e}")
		# In ExportFindings.downloader_ui, after creating download_options_container and checkboxes:
		self.download_options_container.setObjectName("downloadOptionsContainer")
		for cb in self.option_checkboxes:
			cb.setObjectName("downloadOptionCheckBox")

	def get_checked_options(self):
		return [cb.text() for cb in self.option_checkboxes if cb.isChecked()]

	def download_results(self):
		# Collect all checked options
		self.selected_options = self.get_checked_options()
		if not self.selected_options:
			QMessageBox.warning(self.ui, "No Option Selected", "Please select at least one file type to download.")
			return

		options = QFileDialog.Option.DontUseNativeDialog
		base_file_name, _ = QFileDialog.getSaveFileName(
			self.ui, "Save File", "", "All Files (*)", options=options
		)
		if not base_file_name:
			return

		# Remove extension if user typed one
		base_file_name = os.path.splitext(base_file_name)[0]

		# Show a loading message while downloads are in progress
		self.final_download_button.setEnabled(False)  # Disable download button during latency
		QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

		# Perform all downloads
		for opt in self.selected_options:
			if opt == "CSV":
				self.save_as_csv(base_file_name)
			elif opt == "PDF":
				self.save_as_pdf(base_file_name)
			elif opt == "XLSX (Apple Numbers or Microsoft Excel)":
				self.save_as_xlsx(base_file_name)

		# Adaptive latency: ensure at least 3 seconds
		self._download_finalize_timer = QElapsedTimer()
		self._download_finalize_timer.start()
		QTimer.singleShot(100, self.try_finalize_download)

	def try_finalize_download(self):
		try:
			QApplication.processEvents()
			if self._download_finalize_timer.elapsed() >= 3000:
				QApplication.restoreOverrideCursor()
				QMessageBox.information(self.ui, "Download Complete", "Your selected results file(s) have been saved successfully.")
				self.rh.return_home_from_sucess()
			else:
				QTimer.singleShot(100, self.try_finalize_download)
		except Exception as e:
			logging.error(f"Error finalizing download: {e}")
			QMessageBox.critical(self.ui, "Error", f"Error finalizing download: {e}")
			QApplication.restoreOverrideCursor()
			raise RuntimeError(f"Error finalizing download: {e}")

	def save_as_csv(self, file_name):
		try:
			# Assume latest results are at the top of autosave file
			with open(AUTO_SAVE_FILE, "r") as f:
				entries = f.read().strip().split("\n\n---\n\n")
				latest = json.loads(entries[0]) if entries and entries[0].strip() else {}
			df = pd.DataFrame.from_dict(latest, orient='index')
			df.reset_index(inplace=True)
			df.rename(columns={'index': 'Sample'}, inplace=True)
			csv_file = file_name if file_name.endswith('.csv') else file_name + '.csv'
			df.to_csv(csv_file, index=False)
			logging.info(f"Results saved as CSV: {csv_file}")
		except Exception as e:
			logging.error(f"Error saving as CSV: {e}")
			QMessageBox.critical(self.ui, "Error", f"Error saving as CSV: {e}")

	def save_as_xlsx(self, file_name):
		try:
			# Assume latest results are at the top of autosave file
			with open(AUTO_SAVE_FILE, "r") as f:
				entries = f.read().strip().split("\n\n---\n\n")
				latest = json.loads(entries[0]) if entries and entries[0].strip() else {}
			df = pd.DataFrame.from_dict(latest, orient='index')
			df.reset_index(inplace=True)
			df.rename(columns={'index': 'Sample'}, inplace=True)
			xlsx_file = file_name if file_name.endswith('.xlsx') else file_name + '.xlsx'
			df.to_excel(xlsx_file, index=False)
			logging.info(f"Results saved as XLSX: {xlsx_file}")
		except Exception as e:
			logging.error(f"Error saving as XLSX: {e}")
			QMessageBox.critical(self.ui, "Error", f"Error saving as XLSX: {e}")
			raise RuntimeError(f"Error saving as XLSX: {e}")

	def save_as_pdf(self, file_name):
		try:
			# Render the QTableWidget as a styled PDF table, matching QSS, fonts, and data
			table_widget = self.ui.results_table
			if table_widget is None:
				raise RuntimeError("Results table not found.")

			# Extract headers
			headers = [table_widget.horizontalHeaderItem(col).text() for col in range(table_widget.columnCount())]
			vert_headers = [table_widget.verticalHeaderItem(row).text() if table_widget.verticalHeaderItem(row) else str(row+1) for row in range(table_widget.rowCount())]

			# Extract data
			data = []
			for row in range(table_widget.rowCount()):
				row_data = []
				for col in range(table_widget.columnCount()):
					item = table_widget.item(row, col)
					row_data.append(item.text() if item else "")
				data.append(row_data)

			# Compose table data with vertical headers as first column
			table_data = [[""] + headers]
			for row_idx, row_data in enumerate(data):
				table_data.append([vert_headers[row_idx]] + row_data)

			# Prepare PDF file
			pdf_file = file_name if file_name.endswith('.pdf') else file_name + '.pdf'
			doc = SimpleDocTemplate(pdf_file, pagesize=landscape(letter), leftMargin=30, rightMargin=30, topMargin=30, bottomMargin=30)
			elements = []

			# Add title
			styles = getSampleStyleSheet()
			title = Paragraph("LNVSI Tool Results", styles['Title'])
			elements.append(title)
			elements.append(Spacer(1, 24))

			# Table style to mimic QSS/QTableWidget with Arial Black for headers
			table_style = TableStyle([
				('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#d3d3d3")),  # Top header row
				('BACKGROUND', (0, 1), (0, -1), colors.HexColor("#d3d3d3")),  # Left header column
				('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#000000")),
				('TEXTCOLOR', (0, 1), (0, -1), colors.HexColor("#000000")),
				('ALIGN', (0, 0), (-1, -1), 'CENTER'),
				('FONTNAME', (0, 0), (-1, 0), 'Arial-Bold'),  # Top header row
				('FONTNAME', (0, 1), (0, -1), 'Arial-Bold'),  # Left header column
				('FONTNAME', (1, 1), (-1, -1), 'Arial'),      # Table body
				('FONTSIZE', (0, 0), (-1, -1), 10),
				('BOTTOMPADDING', (0, 0), (-1, 0), 12),
				('BOTTOMPADDING', (0, 1), (0, -1), 12),
				('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#888888")),
			])

			# Apply alternating row color for body
			for row in range(2, len(table_data)):
				if (row-1) % 2 == 1:
					table_style.add('BACKGROUND', (1, row), (-1, row), colors.whitesmoke)

			# Make sure Arial Black is available, fallback to Helvetica-Bold if not
			try:
				pdfmetrics.registerFont(TTFont('Arial', 'Arial.ttf'))
				pdfmetrics.registerFont(TTFont('Arial-Bold', 'Arial Black.ttf'))
			except Exception:
				# Fallback to Helvetica-Bold
				table_style.add('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
				table_style.add('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold')
				table_style.add('FONTNAME', (1, 1), (-1, -1), 'Helvetica')

			# Set column widths to fit page width
			col_count = len(table_data[0])
			page_width = landscape(letter)[0] - doc.leftMargin - doc.rightMargin
			col_width = page_width / col_count
			col_widths = [col_width] * col_count

			# Create the table and allow it to split across pages
			table = Table(table_data, colWidths=col_widths, repeatRows=1, repeatCols=1, splitByRow=1)
			table.setStyle(table_style)
			elements.append(table)

			doc.build(elements)
			logging.info(f"Results saved as PDF: {pdf_file}")

		except Exception as e:
			logging.error(f"Error saving as PDF: {e}")
			QMessageBox.critical(self.ui, "Error", f"Error saving as PDF: {e}")
			raise RuntimeError(f"Error saving as PDF: {e}")

class ReturnHome:
	def __init__(self, ui=None):
		self.ui = ui  # Use the passed UI instance directly
		# self.ef will be assigned when needed from the existing instance

	def return_home_from_sucess(self):
		# Properly remove and delete the results_container and all its children
		try:
			if hasattr(self.ui, 'results_container') and isinstance(self.ui.results_container, QWidget):
				for child in self.ui.results_container.findChildren(QWidget):
					child.setParent(None)
					child.deleteLater()
				self.ui.main_layout.removeWidget(self.ui.results_container)
				self.ui.results_container.setParent(None)
				self.ui.results_container.deleteLater()
				del self.ui.results_container

				# Reset the UI to the welcome screen for a new analysis
				self.ui.render_ui()

		except Exception as e:
			logging.error(f"Error clearing results_container: {e}")
			raise RuntimeError(f"Error clearing results_container: {e}")
	
	def return_home_from_error (self):
		# Properly remove and delete the data analysis and all its children
		# (due to a crash)
		# (mid-analysis)
		try:
			if hasattr(self.ui, 'data_analysis_container') and isinstance(self.ui.data_analysis_container, QWidget):
				for child in self.ui.data_analysis_container.findChildren(QWidget):
					child.setParent(None)
					child.deleteLater()
				self.ui.main_layout.removeWidget(self.ui.data_analysis_container)
				self.ui.data_analysis_container.setParent(None)
				self.ui.data_analysis_container.deleteLater()
				del self.ui.data_analysis_container

				# Reset the UI to the welcome screen for a new analysis
				self.ui.render_ui()

		except Exception as e:
			logging.error(f"The worst crash possible for this tool has occured!: {e}")
			QMessageBox.critical(
			"If you are seeing this message, then this app's repo has been severely" 
			"corrupted. \n Forget support and call Luke directly at 510-856-6002 to" 
			"quickly resolve this issue for the other scientists affected. " \
			"Rollback to previous version of the app will likely provide a temporary fix."
			)
			raise RuntimeError(f"ConoKillerBug: {e}")
	
	def return_home_from_cancel(self):
	
		# Properly remove and delete the data analysis container and all its children 
		# (due to a manual cancel)
		# (mid-anaysis)
		try:
			if hasattr(self.ui, 'data_analysis_container') and isinstance(self.ui.data_analysis_container, QWidget):
				for child in self.ui.data_analysis_container.findChildren(QWidget):
					child.setParent(None)
					child.deleteLater()
				self.ui.main_layout.removeWidget(self.ui.data_analysis_container)
				self.ui.data_analysis_container.setParent(None)
				self.ui.data_analysis_container.deleteLater()
				del self.ui.data_analysis_container

				# Reset the UI to the welcome screen for a new analysis
				self.ui.render_ui()

		except Exception as e:
			logging.error(f"The worst crash possible for this tool has occured!: {e}")
			QMessageBox.critical(
			"If you are seeing this message, then this app's repo has been severely" 
			"corrupted. \n Forget support and call Luke directly at 510-856-6002 to" 
			"quickly resolve this issue for the other scientists affected. " \
			"Rollback to previous version of the app will likely provide a temporary fix."
			)
			raise RuntimeError(f"ConoKiller Bug: {e}")
	
	def return_home_from_discard(self):
		# Properly remove and delete the results container and all its children 
		# (due to a manual results discard) 
		# (post-analysis ; a bit more complex due to the json entry needing to be dsicarded)
		# We discard the json entry by treating as if it was an error
		# (but without true errors in logs)
		try:
			# Remove and delete the results_container and all its children (post-analysis discard)
			if hasattr(self.ui, 'results_container') and isinstance(self.ui.results_container, QWidget):
				for child in self.ui.results_container.findChildren(QWidget):
					child.setParent(None)
					child.deleteLater()
				self.ui.main_layout.removeWidget(self.ui.results_container)
				self.ui.results_container.setParent(None)
				self.ui.results_container.deleteLater()
				del self.ui.results_container

			# Discard the latest autosave entry (remove the top entry from AUTO_SAVE_FILE)
			try:
				if os.path.exists(AUTO_SAVE_FILE):
					with open(AUTO_SAVE_FILE, "r") as f:
						entries = f.read().strip().split("\n\n---\n\n")
					# Remove the first (latest) entry if present
					# This way, the json shows that nothing ever happened
					if entries and entries[0].strip():
						entries = entries[1:]
						with open(AUTO_SAVE_FILE, "w") as f:
							for i, entry in enumerate(entries):
								if entry.strip():
									f.write(entry)
									if i < len(entries) - 1:
										f.write("\n\n---\n\n")
			except Exception as e:
				logging.warning(f"Could not discard autosave entry: {e}")

			# Reset the UI to the welcome screen for a new analysis
			self.ui.render_ui()

		except Exception as e:
			logging.error(f"The worst crash possible for this tool has occured!: {e}")
			QMessageBox.critical(
			"If you are seeing this message, then this app's repo has been severely" 
			"corrupted. \n Forget support and call Luke directly at 510-856-6002 to" 
			"quickly resolve this issue for the other scientists affected. " \
			"Rollback to previous version of the app will likely provide a temporary fix."
			)
			raise RuntimeError(f"ConoKiller Bug: {e}")

def main():
    import sys
    app = QApplication(sys.argv)
    window = UI()
    window.render_ui()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
