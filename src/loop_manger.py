# Basic Imports
import logging
from typing import TYPE_CHECKING

# PyQt6 GUI Imports
from PyQt6.QtWidgets import (
    QApplication
)

# Core Imports
from config import LOG_FILE

if TYPE_CHECKING:
    from main import UI

# Connect to the global logging file
global_logging_file = LOG_FILE

logging.basicConfig(
    filename= global_logging_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d'
)

class ReturnHome:
    def __init__(self, ui: object = None):
        self.ui = ui  # UI is an instance of the UI (QMainWindow) class
    

    def _full_reset(self):
        # Remove all widgets and layouts from the main window, clear cached results, and re-render UI

        # Remove all widgets from main_layout
        if hasattr(self.ui, 'main_layout'):
            while self.ui.main_layout.count():
                item = self.ui.main_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()
        # Remove references to containers and layouts
        for attr in [
            'welcome_container', 'welcome_layout', 'sub_welcome_label',
            'results_container', 'results_layout', 'results_label', 'results_line', 'results_portal',
            'results_portal_layout', 'results_table', 'download_button', 'discard_button', 'finish_button',
            '_cached_results', '_active_gif_containers', '_finalize_timer', '_upload_timer', '_upload_msgbox'
        ]:
            if hasattr(self.ui, attr):
                setattr(self.ui, attr, None)

        # Remove any background label
        if hasattr(self.ui, 'background_label') and self.ui.background_label is not None:
            self.ui.background_label.setParent(None)
            self.ui.background_label.deleteLater()
            self.ui.background_label = None
        # Reset state in AnalysisEngine and ExportFindings
        if hasattr(self.ui, 'ae'):
            self.ui.ae.filename = None
            self.ui.ae.wavelengths = None
            self.ui.ae.absorbance = None
            self.ui.ae.positive_control = None

            self.ui.ae.sample_names = []
            self.ui.ae.num_analytes = 0
            self.ui.ae.processed = None
        if hasattr(self.ui, 'ef'):
            if hasattr(self.ui.ef, '_export_table_data'):
                self.ui.ef._export_table_data = None
            if hasattr(self.ui.ef, '_export_table_headers'):
                self.ui.ef._export_table_headers = None

        # Re-render the UI from scratch
        self.ui.render_ui()
        QApplication.processEvents()

    def return_home_from_sucess(self):
        self._full_reset()
        logging.info("Returning home from success state.")

    def return_home_from_error(self):
        self._full_reset()
        logging.info("Returning home from error state.")

    def return_home_from_cancel(self):
        self._full_reset()
        logging.info("Returning home from cancel state.")

    def return_home_from_discard(self):
        self._full_reset()
        logging.info("Returning home from discard state.")
