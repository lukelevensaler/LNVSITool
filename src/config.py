""" 
Essential configuration logic for the whole application.
"""

# Import necessary modules
import os
import logging
import warnings
from PyQt6.QtCore import QUrl

# Set up project-relative utils dir (user-writable, not system dir)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Place all LNVSI Tool Utilities in the user's home directory (~/LNVSI_Tool_Utilities)
user_home = os.path.expanduser("~")
utilsfolder = os.path.join(user_home, "LNVSI Tool Utilities")

if not os.path.exists(utilsfolder):
    os.makedirs(utilsfolder)

# Set up logging (still in utilsfolder)
log_dir = os.path.join(utilsfolder, "Logs")
LOG_FILE = os.path.join(log_dir, "LNVSI Tool.log") # CALLED IN OTHER MODULES

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w') as f:
        pass

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s - line %(lineno)d'
)

# Redirect all warnings (including OptimizeWarning, DeprecationWarning, etc.) to the logging system
logging.captureWarnings(True)

# Optionally, customize warnings to always show and log with stack info
warnings.simplefilter('always')

# Add a handler for uncaught exceptions to log them with traceback
import sys

def log_uncaught_exceptions(exctype, value, tb):
    import traceback
    logging.critical("Uncaught exception:", exc_info=(exctype, value, tb))
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = log_uncaught_exceptions

# Set up autosave
autosave_dir = os.path.join(utilsfolder, "Autosaves")
if not os.path.exists(autosave_dir):
    os.makedirs(autosave_dir)
AUTO_SAVE_RESULTS_FILE = os.path.join(autosave_dir, "autosave_results.json") # CALLED IN OTHER MODULES

# Asset paths using project-relative assets directory
assets_dir = os.path.join(project_root, "assets")

def resource_path(filename): # CALLED IN OTHER MODULES
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        # Running in a PyInstaller bundle
        return os.path.join(meipass, "assets", filename)
    else:
        # Running in development
        return os.path.join(assets_dir, filename)

def qfiledialog__pinned_locations(): # CALLED IN OTHER MODULES 
    """ Get and pin the locations for QFileDialog instances. """
    pinned_locations = [
        user_home,
        os.path.join(os.path.expanduser("~"), "Documents"),  # Documents 
        os.path.join(os.path.expanduser("~"), "Downloads"),  # Downloads
        os.path.join(os.path.expanduser("~"), "Desktop"),  # Desktop
    ]
    return [QUrl.fromLocalFile(path) for path in pinned_locations]