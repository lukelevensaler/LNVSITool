""" 

Main entry point for LNVSI Tool and frontend GUI.

"""

# Basic Imports
import os
import logging

# PyQt6 GUI Imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton,
    QMessageBox, QFrame, QProgressBar, QTableWidget, QTableWidgetItem, 
    QHeaderView
)
from PyQt6.QtGui import QIcon, QMovie, QPixmap, QCursor
from PyQt6.QtCore import QTimer, QSize, Qt

# Core Imports 
from config import (
    LOG_FILE, resource_path
)
from analysis import AnalysisEngine
from utils import ExportFindings, ErrorManager
from loop_manger import ReturnHome

# Connect to the global logging file
global_logging_file = LOG_FILE

logging.basicConfig(
    filename= global_logging_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d'
)


class UI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ConoWare Levensaler Neogastropod Venomic Similarity Indexing Tool (version: alpha-testing)")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(1000, 800)
       
        self.icon_path = resource_path("icon.png")
        self.background_path = resource_path("background.png")
        self.gif_path = resource_path("countdown.gif")
        
        # Set window icon
        if os.path.exists(self.icon_path):
            self.setWindowIcon(QIcon(self.icon_path))
        else:
            logging.error(f"Icon file not found: {self.icon_path}")
            raise FileNotFoundError(f"Icon file not found: {self.icon_path}")
        
        self.setup_background()

        # Set central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)  # Set the central widget
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.central_widget.setLayout(self.main_layout)

        # Other basic initializations - create in correct order
        # First create ReturnHome since other classes need it
        self.rh = ReturnHome(self)
        # Then create the other classes
        self.ae = AnalysisEngine(self)
        self.ef = ExportFindings(self)
        
        # Initialize progress_bar as None until it's created
        self.progress_bar = None
        self.splash_text = "Initializing..."  # Ensure splash_text always exists
        self._analysis_cursor_forced = False  # Track if wait cursor is forced
        self._showing_results = False  # Guard to prevent duplicate/flickering results UI        

    def setup_background(self):
        # Set background image
        try:
            if os.path.exists(self.background_path):
                if not hasattr(self, 'background_label') or self.background_label is None:
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

    def resizeEvent(self, event):

        """
        Ensure the background image always expands to fit the window, including on resize/fullscreen.
        """
        super().resizeEvent(event)
        if hasattr(self, 'background_label'):
            self.background_label.setGeometry(0, 0, self.width(), self.height())

    def render_ui(self):

        self.setup_background()
        # Set welcome_container to be centered with margins (middle third of window)
        self.welcome_container = QWidget(self)
        self.main_layout.addWidget(
            self.welcome_container, 
            alignment=
            Qt.AlignmentFlag.AlignHCenter | 
            Qt.AlignmentFlag.AlignVCenter
        )

        self.welcome_layout = QVBoxLayout(self.welcome_container)
        self.welcome_container.setLayout(self.welcome_layout)

        # Main welcome label (HTML Rich Text))(top-aligned, wide)
        self.welcome_label = QLabel(
            '<span class="main-title">Welcome to the LNVSI Tool (version alpha-testing)!</span><br>'
            '<span class="subtitle">Created by Luke Levensaler, 2025</span>'
        )
        self.welcome_label.setObjectName("welcomeLabel")
        self.welcome_label.setTextFormat(Qt.TextFormat.RichText)
        self.welcome_label.setMinimumWidth(700)
        self.welcome_label.setMaximumWidth(1200)
        self.welcome_label.setMinimumHeight(120)
        self.welcome_label.setMaximumHeight(220)
        self.welcome_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.welcome_layout.addSpacing(12)
        self.welcome_layout.addWidget(
            self.welcome_label,
            alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        self.welcome_layout.addSpacing(8)

        # Sub welcome label (spaced below welcome label, above upload button)
        self.sub_welcome_label = QLabel(
            "Let's analyze your Levensaler Assay-derived venomic datasets with the "
            "Levensaler Neogastropod Venomic Similarity Index's (LNVSI) machine learning-powered "
            "statistical algorithm! The results will tell you if they share similarities to your "
            "conotoxin positive control.\n"
            "If your device can handle it (see the ConoWare Project's website or our GitHub page) "
            "and your peptidomes are conopeptide-like, "
            "you should install ConoBot AI to help you identify your proteomes' unique "
            "primary structure cysteine frameworks!"
        )
        self.sub_welcome_label.setObjectName("subWelcomeLabel")
        self.sub_welcome_label.setMinimumWidth(400)
        self.sub_welcome_label.setMaximumWidth(900)
        self.sub_welcome_label.setMinimumHeight(300)
        self.sub_welcome_label.setMaximumHeight(500)
        self.sub_welcome_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.sub_welcome_label.setWordWrap(True)
        self.welcome_layout.addWidget(
            self.sub_welcome_label,
            alignment=
            Qt.AlignmentFlag.AlignHCenter | 
            Qt.AlignmentFlag.AlignVCenter
        )

        # Upload CSV Button (bottom, rounded corners via QSS)
        self.upload_button = QPushButton("Upload Your CSV File Here")
        self.upload_button.setObjectName("uploadButton")
        self.upload_button.setMinimumWidth(220)
        self.upload_button.setMaximumWidth(340)
        self.upload_button.setMinimumHeight(60)
        self.upload_button.setMaximumHeight(80)
        self.upload_button.clicked.connect(self.confirm_csv_upload)
        self.welcome_layout.addWidget(
            self.upload_button,
            alignment=
            Qt.AlignmentFlag.AlignHCenter | 
            Qt.AlignmentFlag.AlignBottom
        )
        self.upload_button.setEnabled(True)  # Always ensure enabled on UI render

        # Set welcome screen object names for QSS styling
        self.central_widget.setObjectName("mainCentralWidget")
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
                if self.upload_button is not None:
                    try:
                        self.upload_button.setEnabled(False)  # Disable upload button during latency
                    except Exception:
                        pass
                # Do NOT show countdown gif/timer here. Only show after file is selected in upload_csv.
                self.ae.upload_csv()
            else:
                # User aborted, just close the dialog and do nothing
                if self.upload_button is not None:
                    try:
                        self.upload_button.setEnabled(True)
                    except Exception:
                        pass
                self.remove_countdown_gif_and_timer()  # Remove any active gif/timer if present
                return
        except Exception as e:
            logging.error(f"Error confirming CSV upload: {e}")
            QMessageBox.critical(self, "Error", f"Error confirming CSV upload: {e}")
            if self.upload_button is not None:
                try:
                    self.upload_button.setEnabled(True)
                except Exception:
                    pass
            raise RuntimeError(f"Error confirming CSV upload: {e}")

    def launch_data_analysis_mini_screen(self):
        # Remove any active countdown gif/timer before proceeding
        self.remove_countdown_gif_and_timer()
        # Force wait cursor for analysis stage
        self.force_wait_cursor()
        # Remove and delete the welcome_container (simple/original logic)
        if hasattr(self, 'welcome_container') and self.welcome_container is not None:
            self.main_layout.removeWidget(self.welcome_container)
            self.welcome_container.setParent(None)
            self.welcome_container.deleteLater()
            self.welcome_container = None    

        # Data analysis container: white, rounded corners, no border
        self.data_analysis_container = QWidget(self)
        self.data_analysis_container.setVisible(True)
        self.data_analysis_container.setObjectName("dataAnalysis")
        self.data_analysis_container.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.data_analysis_container, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.data_analysis_container.setFixedSize(700, 480)
        self.data_analysis_layout = QVBoxLayout(self.data_analysis_container)

        # Add a top label with breathing room and centered text
        self.data_analysis_label = QLabel("Running Similarity Engine...")
        # Font via QSS only
        self.data_analysis_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.data_analysis_label.setContentsMargins(0, 30, 0, 20)  # Top and bottom breathing room
        self.data_analysis_layout.addWidget(self.data_analysis_label, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

        # Add a horizontal line (QFrame) after the label for separation (always add here)
        self.line = QFrame()
        self.line.setObjectName("analysisLine")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        # Style via QSS only
        self.data_analysis_layout.addWidget(self.line)

        # Centered Perfectly Progress Bar Container
        self.progress_container = QWidget(self.data_analysis_container)
        self.progress_container.setFixedHeight(140)
        self.progress_container.setFixedWidth(400)
        self.progress_container.setContentsMargins(0, 0, 0, 0) # No margins for the progress bar
        # Style via QSS only
        self.data_analysis_layout.addStretch(1)
        self.data_analysis_layout.addWidget(
            self.progress_container, 
            alignment=
            Qt.AlignmentFlag.AlignHCenter
            | Qt.AlignmentFlag.AlignVCenter
        )
        self.data_analysis_layout.addStretch(1)
       
        # Progress Bar Layout
        self.progress_layout = QVBoxLayout(self.progress_container)
        self.progress_layout.setContentsMargins(0, 0, 0, 0) # No margins for the progress bar
        self.progress_layout.setSpacing(0)

        # Add a label for the progress bar (top 1/2 of progress layout)
        # Ensure analysis_progress_bar is created before set_splash_text is called
        self.progress_label = QLabel(self.splash_text)
        self.progress_label.setObjectName("progressLabel")

        # Font via QSS only
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)
        self.progress_label.setFixedHeight(40)
        self.progress_layout.addWidget(self.progress_label, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)

        self.analysis_progress_bar = QProgressBar(self.progress_container)
        self.analysis_progress_bar.setRange(0, 100)
        self.analysis_progress_bar.setValue(0)
        self.analysis_progress_bar.setFixedHeight(55)
        self.analysis_progress_bar.setFixedWidth(300)
        # Style via QSS only

        # Set progress_bar reference for compatibility
        self.progress_bar = self.analysis_progress_bar

        # Add progress bar and gif in a horizontal layout to prevent overlap
        self.progress_bar_row = QHBoxLayout()
        self.progress_bar_row.setContentsMargins(0, 0, 0, 0)
        self.progress_bar_row.setSpacing(0)
        self.progress_bar_row.addWidget(self.analysis_progress_bar, alignment=Qt.AlignmentFlag.AlignHCenter)
        # Always show countdown gif/timer at analysis start, and ensure it's visible
        self.show_countdown_gif(self.progress_bar_row, self.progress_container, trigger_type='progress', duration_ms=3000)
        self.progress_layout.addLayout(self.progress_bar_row)

        # Cancel button at bottom left of entire container
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.setContentsMargins(0, 30, 0, 20)
        self.cancel_button.setFixedSize(250, 60)
       
        self.cancel_button.setMinimumWidth(220)
        self.cancel_button.setMaximumWidth(340)
        self.cancel_button.setMinimumHeight(60)
        self.cancel_button.setMaximumHeight(80)
        self.data_analysis_layout.addWidget(
            self.cancel_button,
            alignment=
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom
        )
        self.cancel_button.clicked.connect(self.confirm_analysis_cancellation)
        self.cancel_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
       
        # Data analysis container object names for QSS styling
        self.data_analysis_container.setObjectName("dataAnalysis")
        self.data_analysis_label.setObjectName("analysisLabel")
        self.progress_container.setObjectName("progressContainer")
        self.progress_label.setObjectName("progressLabel")
        self.analysis_progress_bar.setObjectName("progressBar")
        self.cancel_button.setObjectName("cancelButton")
        self.set_all_button_cursors()

    def set_splash_text(self):
        # Set the splash text based on the progress
        progress = self.analysis_progress_bar.value()
        splash = ""
        if progress == 0:
            splash = "Initializing..."
            self._showing_results = False  # Reset guard at start of analysis
        elif 0 < progress < 10:
            splash = "Preparing Data..."
        elif 10 <= progress < 20:
            splash = "Reading CSV File..."
        elif 20 <= progress < 25:
            splash = "Preprocessing Curves..."
        elif 25 <= progress < 30:
            splash = "Applying Lowpass Filter..."
        elif 30 <= progress < 35:
            splash = "Smoothing Data..."
        elif 35 <= progress < 40:
            splash = "Curve Smoothing Complete."
        elif 40 <= progress < 50:
            splash = "Segmenting Spectral Regions..."
        elif 50 <= progress < 60:
            splash = "Deconvolving Voigt Profiles..."
        elif 60 <= progress < 65:
            splash = "Aligning Curves (DTW)..."
        elif 65 <= progress < 70:
            splash = "Computing Similarity Metrics..."
        elif 70 <= progress < 75:
            splash = "Determining Composite Similarity Scores..."
        elif 75 <= progress < 80:
            splash = "Running Statistical Tests..."
        elif 80 <= progress < 90:
            splash = "Analyzing All Samples..."
        elif 90 <= progress < 95:
            splash = "Saving Results..."
        elif 95 <= progress < 100:
            splash = "Finalizing Analysis..."
        elif progress == 100:
            splash = "Algorithm Returned Valid Results!..."
            # Only show results screen once
            if not self._showing_results:
                self._showing_results = True
                self.remove_countdown_gif_and_timer()
                self.restore_cursor()  # Restore cursor when analysis is done
                self.results()
        # Always update the progress label
        if hasattr(self, 'progress_label'):
            self.progress_label.setText(splash)
        return splash

    def results(self):
        # Prevent recursive or repeated results UI
        if self._showing_results and hasattr(self, 'results_container') and self.results_container is not None:
            return
        self._showing_results = True
        # Restore native cursor when analysis is done and results are shown
        QApplication.restoreOverrideCursor()

        # Remove and delete any existing results_container (simple/original logic)
        if hasattr(self, 'results_container') and self.results_container is not None:
            self.main_layout.removeWidget(self.results_container)
            self.results_container.setParent(None)
            self.results_container.deleteLater()
            self.results_container = None

        # Remove and delete the data_analysis container (simple/original logic)
        if hasattr(self, 'data_analysis_container') and self.data_analysis_container is not None:
            self.main_layout.removeWidget(self.data_analysis_container)
            self.data_analysis_container.setParent(None)
            self.data_analysis_container.deleteLater()
            self.data_analysis_container = None
        
        # Create a new container for the results with same dimensions as previous two containers
        self.results_container = QWidget(self)
        self.results_container.setObjectName("resultsContainer")
        self.main_layout.addWidget(self.results_container, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.results_container.setFixedSize(700, 480)
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
        self.results_container.setContentsMargins(0, 0, 0, 0)
        self.results_layout = QVBoxLayout(self.results_container)

        # Add a top label for the results screen, centered with breathing room and QSS dash style
        self.results_label = QLabel("Algorithm Output - Results")
        self.results_label.setObjectName("resultsLabel")
        self.results_label.setFixedHeight(50)
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.results_layout.addWidget(self.results_label)

        # Add a horizontal line (QFrame) after the label for separation
        self.results_line = QFrame()
        self.results_line.setObjectName("resultsLine")
        self.results_line.setFrameShape(QFrame.Shape.HLine)
        self.results_line.setFrameShadow(QFrame.Shadow.Sunken)
        self.results_layout.addWidget(self.results_line)

        # Results portal (container for the table)
        self.results_portal = QWidget(self.results_container)
        self.results_portal.setObjectName("resultsWidget")
        self.results_portal.setContentsMargins(0, 0, 0, 0)
        self.results_portal_layout = QVBoxLayout(self.results_portal)
        self.results_portal.setLayout(self.results_portal_layout)
        self.results_layout.addWidget(self.results_portal)

        # Results table
        self.results_table = QTableWidget(self.results_portal)
        self.results_table.setObjectName("resultsTable")
        self.results_table.setFixedSize(500, 500)

        # Center the table widget in the results_portal layout
        self.results_portal_layout.addWidget(self.results_table, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.results_table.setContentsMargins(0, 0, 0, 0)
        self.results_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.results_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.results_table.setColumnCount(4)
        self.results_table.setRowCount(self.ae.num_analytes)
        self.results_table.setHorizontalHeaderLabels(["Sample", "Conotoxin Like?", "Similarity %", "p-value"])
        self.results_table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_table.verticalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Set resize mode for all columns to ResizeToContents for best fit
        for col in range(4):
            self.results_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionsMovable(False)
        self.results_table.verticalHeader().setSectionsMovable(False)
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
        self.results_table.horizontalHeader().setSectionsClickable(False)
        self.results_table.verticalHeader().setSectionsClickable(False)

        # Populate the table with results for each analyte (excluding positive control)
        if hasattr(self.ae, 'sample_names') and hasattr(self.ae, 'analyze_all_samples'):
            # Cache results in self for reuse
            if not hasattr(self, '_cached_results') or self._cached_results is None:
                self._cached_results = self.ae.analyze_all_samples()
            results_dict = self._cached_results
            for row, sample in enumerate(self.ae.sample_names):
                result = results_dict.get(sample, {})
                # Ensure conotoxin_like is interpreted as boolean, not string or number
                conotoxin_like = result.get("conotoxin_like", False)
                if isinstance(conotoxin_like, str):
                    conotoxin_like = conotoxin_like.strip().lower() == "true"
                conotoxin_like = bool(conotoxin_like)
                # Fill table with correct values
                for col, value in enumerate([
                    str(sample),
                    "Yes" if conotoxin_like else "No",
                    f"{result.get('similarity_percent', 0):.2f}",
                    f"{result.get('p_value', 1):.3g}"
                ]):
                    item = QTableWidgetItem(value)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
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

        # Download and Discard buttons on the same horizontal plane
        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 30, 0, 0)
        button_row.setSpacing(24)
        self.download_button = QPushButton("Select Method of Exporting Results")
        self.download_button.setObjectName("downloadUILaunchButton")
        self.download_button.setMinimumWidth(220)
        self.download_button.setMaximumWidth(340)
        self.download_button.setMinimumHeight(60)
        self.download_button.setMaximumHeight(80)
        self.download_button.clicked.connect(self.confirm_results_download)
        button_row.addWidget(self.download_button, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        self.download_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.discard_button = QPushButton("Discard Results")
        self.discard_button.setObjectName("discardButton")
        self.discard_button.setMinimumWidth(220)
        self.discard_button.setMaximumWidth(340)
        self.discard_button.setMinimumHeight(60)
        self.discard_button.setMaximumHeight(80)
        self.discard_button.clicked.connect(self.confirm_results_discard)
        button_row.addWidget(self.discard_button, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        self.discard_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.results_layout.addLayout(button_row)
        self.set_all_button_cursors()
        # At the end of results(), reset the guard so user can re-run analysis if needed
        self._showing_results = False

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
                if self.download_button is not None:
                    try:
                        self.download_button.setEnabled(False)  # Disable download button during latency
                    except Exception:
                        pass
                self.restore_cursor()  # Restore cursor before file dialog
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
                if self.discard_button is not None:
                    try:
                        self.discard_button.setEnabled(False)  # Disable discard button during latency
                    except Exception:
                        pass
                self.restore_cursor()  # Restore cursor before returning home
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
                if self.cancel_button is not None:
                    try:
                        self.cancel_button.setEnabled(False)  # Disable cancel button during latency
                    except Exception:
                        pass
                self.restore_cursor()  # Restore cursor before returning home
                self.rh.return_home_from_cancel()
                ErrorManager.SetErrorsPaused(True)  # Pause error manager on cancel
            else:
                # User aborted, just close the dialog and do nothing
                return
        except Exception as e:
            logging.error(f"Error confirming cancel: {e}")
            QMessageBox.critical(self, "Error", f"Error confirming cancel: {e}")
            raise RuntimeError(f"Error confirming cancel: {e}")
    
    def show_countdown_gif(self, parent_layout, parent_container, trigger_widget=None, trigger_type='progress', duration_ms=None):
        
        # Always remove any existing gif/timer before adding a new one
        
        try:
            self.remove_countdown_gif_and_timer()
        except Exception:
            pass
        gif_size = 55
        timer_width = 60
        timer_height = 28
        timer_margin_bottom = 6
        group_margin_bottom = 36

        try:
            group_widget = QWidget(parent_container)
            group_layout = QVBoxLayout(group_widget)
            group_layout.setContentsMargins(0, 0, 0, group_margin_bottom)
            group_layout.setSpacing(timer_margin_bottom)

            # Timer label (styled via QSS only)
            timer_label = QLabel(group_widget)
            timer_label.setFixedSize(timer_width, timer_height)
            timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            timer_label.setObjectName("countdownTimerLabel")
            timer_label.hide()
            group_layout.addWidget(timer_label, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)

            # Gif container and label
            gif_container = QWidget(group_widget)
            gif_container.setFixedSize(gif_size, gif_size)
            gif_layout = QHBoxLayout(gif_container)
            gif_layout.setContentsMargins(0, 0, 0, 0)
            gif_layout.setSpacing(0)
            gif_label = QLabel(gif_container)
            gif_label.setFixedSize(gif_size, gif_size)
            gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            gif_layout.addWidget(gif_label)
            group_layout.addWidget(gif_container, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

            # Load gif if available
            movie = None
            if os.path.exists(self.gif_path):
                movie = QMovie(self.gif_path)
                movie.setScaledSize(QSize(gif_size, gif_size))
                gif_label.setMovie(movie)
                movie.jumpToFrame(0)
                movie.start()
            else:
                logging.error(f"Countdown gif not found: {self.gif_path}")

            # Placement logic for upload/download/cancel/discard buttons
            valid_triggers = [getattr(self, 'upload_button', None), getattr(self, 'download_button', None), getattr(self, 'cancel_button', None), getattr(self, 'discard_button', None)]
            trigger_ok = False
            if trigger_widget is not None and trigger_widget in valid_triggers:
                try:
                    trigger_ok = trigger_widget.isWidgetType() and trigger_widget.isVisible()
                except Exception:
                    trigger_ok = False
            parent_layout_ok = False
            if parent_layout is not None and hasattr(parent_layout, 'indexOf') and trigger_ok:
                try:
                    index = parent_layout.indexOf(trigger_widget)
                    if index != -1:
                        parent_layout.insertWidget(index, group_widget, alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
                        trigger_widget.setContentsMargins(0, 0, 0, group_margin_bottom)
                        group_widget.setContentsMargins(0, 0, 0, group_margin_bottom)
                    else:
                        parent_layout.addWidget(group_widget, alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
                    parent_layout_ok = True
                except Exception:
                    parent_layout_ok = False
            if not parent_layout_ok:
                try:
                    if parent_layout is not None and (isinstance(parent_layout, QHBoxLayout) or isinstance(parent_layout, QVBoxLayout)):
                        parent_layout.addWidget(group_widget, alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
                except Exception:
                    pass

            # Countdown logic (non-blocking, keeps UI responsive)
            if duration_ms is not None and duration_ms > 0:
                timer_label.show()
                seconds_left = duration_ms // 1000
                timer_label.setText(f"{seconds_left}s")
                def update_timer():
                    nonlocal seconds_left
                    seconds_left -= 1
                    try:
                        if timer_label and hasattr(timer_label, 'isVisible') and timer_label.isVisible():
                            if seconds_left > 0:
                                timer_label.setText(f"{seconds_left}s")
                                QTimer.singleShot(1000, update_timer)
                            else:
                                timer_label.setText("")
                                timer_label.hide()
                    except Exception:
                        return
                QTimer.singleShot(1000, update_timer)

            # Store reference for later removal if needed
            group_widget._timer_label = timer_label
            group_widget._gif_label = gif_label
            group_widget._movie = movie
            group_widget._parent_layout = parent_layout
            group_widget._parent_container = parent_container
            self._active_gif_containers = [group_widget]  # Only one at a time
        except Exception:
            pass

    def remove_countdown_gif_and_timer(self):
        # Remove all active gif containers and their timer labels
        if hasattr(self, '_active_gif_containers'):
            for gif_container in self._active_gif_containers:
                try:
                    if gif_container is not None:
                        if hasattr(gif_container, '_parent_layout') and gif_container._parent_layout is not None and hasattr(gif_container._parent_layout, 'removeWidget'):
                            try:
                                gif_container._parent_layout.removeWidget(gif_container)
                            except Exception:
                                pass
                        try:
                            gif_container.setParent(None)
                        except Exception:
                            pass
                        try:
                            gif_container.deleteLater()
                        except Exception:
                            pass
                except Exception:
                    pass
            try:
                self._active_gif_containers.clear()
            except Exception:
                pass
        self._active_gif_containers = []

    def close_all_message_boxes(self):
        # Only close real QMessageBox instances (not StandardButton results)
        if hasattr(self, '_upload_msgbox') and self._upload_msgbox and hasattr(self._upload_msgbox, 'isVisible') and self._upload_msgbox.isVisible():
            self._upload_msgbox.close()
        # Do not check _warning_msgbox, as static QMessageBox.warning returns a button, not a widget

    def set_all_button_cursors(self):
        # Set native pointer cursor for all clickable buttons
        for btn_name in [
            'upload_button', 'download_button', 'discard_button', 'cancel_button',
            'confirm_button', 'retry_button', 'return_button', 'close_button',
        ]:
            btn = getattr(self, btn_name, None)
            try:
                if btn is not None and hasattr(btn, 'setCursor'):
                    # Use the native system arrow cursor for buttons
                    btn.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            except Exception:
                continue

    def force_wait_cursor(self):
        if not self._analysis_cursor_forced:
            # Use the native system wait/spinning wheel cursor (platform-appropriate)
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
            self._analysis_cursor_forced = True

    def restore_cursor(self):
        if self._analysis_cursor_forced:
            QApplication.restoreOverrideCursor()
            self._analysis_cursor_forced = False

def main():
    import sys
    app = QApplication(sys.argv)

    # Load and globally apply styles.qss to ALL UI everywhere
    from config import resource_path
    qss_path = resource_path("styles.qss")
    if os.path.exists(qss_path):
        with open(qss_path, "r") as f:
            styles = f.read()
        app.setStyleSheet(styles)  # Apply globally to all widgets
    window = UI()
    window.render_ui()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()