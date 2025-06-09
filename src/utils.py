# Basic Imports
import logging
import csv

# Data Science Imports
import pandas as pd
from statsmodels.stats.multitest import multipletests

# PyQt6 GUI Imports
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QFileDialog, 
    QMessageBox
)
from PyQt6.QtCore import Qt

# ReportLab Imports (for PDF generation from results)
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Core Imports
from config import qfiledialog__pinned_locations, LOG_FILE

# Connect to the global logging file
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d'
)

class FDRUtils:
    """
    Utility class for FDR correction.
    """

    @staticmethod
    def fdr_correction(pvals, alpha=0.05, method='fdr_bh'):
        """
        Apply FDR correction to a list of p-values.
        Returns a tuple (rejected, pvals_corrected).
        """
        rejected, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method=method)
        return rejected, pvals_corrected

class ExportFindings:
    
    def __init__(self, ui=None):
        self.ui = ui  # Use the passed UI instance directly

    def downloader_ui(self):
        try:
            # Remove and delete the results_portal and all its children
            if hasattr(self.ui, 'results_portal') and self.ui.results_portal is not None:
                for child in self.ui.results_portal.findChildren(QWidget):
                    if child is not None:
                        child.setParent(None)
                        child.deleteLater()
                if self.ui.results_layout is not None:
                    self.ui.results_layout.removeWidget(self.ui.results_portal)
                if self.ui.results_portal is not None:
                    self.ui.results_portal.setParent(None)
                    self.ui.results_portal.deleteLater()
                    del self.ui.results_portal
        except Exception as e:
            logging.error(f"Error removing results_portal: {e}")
            QMessageBox.critical(self.ui, "Error", f"Error removing results portal: {e}")
            return

            # Save a snapshot of the table data before deleting the table widget
        self._export_table_data = None
        self._export_table_headers = None
        if hasattr(self.ui, 'results_table') and self.ui.results_table is not None:
            table = self.ui.results_table
            headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
            data = []
            for row in range(table.rowCount()):
                row_data = []
                for col in range(table.columnCount()):
                    item = table.item(row, col)
                    row_data.append(item.text() if item else "")
                data.append(row_data)
            self._export_table_headers = headers
            self._export_table_data = data

        # Create a new results_portal widget for export options
        margin_top_bottom = 60
        margin_sides = int(self.ui.width() * 1/3)
        container_width = self.ui.width() - 2 * margin_sides
        container_height = self.ui.height() - 2 * margin_top_bottom
        container_x = margin_sides
        container_y = margin_top_bottom
        self.ui.results_portal = QWidget(self.ui.results_container)
        self.ui.results_portal.setGeometry(
            container_x,
            container_y,
            container_width,
            container_height
        )
        self.ui.results_portal.setContentsMargins(0, 50, 50, 0)
        self.ui.results_portal_layout = QVBoxLayout(self.ui.results_portal)
        self.ui.results_portal.setLayout(self.ui.results_portal_layout)
        self.ui.results_layout.insertWidget(1, self.ui.results_portal)

        # Export options label
        export_label = QLabel("Select Export Format:")
        export_label.setObjectName("exportLabel")
        export_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.ui.results_portal_layout.addWidget(export_label)

        # Export buttons row
        export_row = QHBoxLayout()
        export_row.setSpacing(24)
        btn_csv = QPushButton("Export as CSV")
        btn_pdf = QPushButton("Export as PDF")
        btn_xlsx = QPushButton("Export as XLSX")
        for btn in [btn_csv, btn_pdf, btn_xlsx]:
            btn.setFixedSize(200, 60)
            export_row.addWidget(btn)
        self.ui.results_portal_layout.addLayout(export_row)

        # Connect export options buttons
        btn_csv.clicked.connect(lambda: self.export_results('csv'))
        btn_pdf.clicked.connect(lambda: self.export_results('pdf'))
        btn_xlsx.clicked.connect(lambda: self.export_results('xlsx'))

        # Set styles for export options buttons
        for btn in [btn_csv, btn_pdf, btn_xlsx]:
            btn.setObjectName("exportButton")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

    def export_results(self, fmt):
        try:
            headers = getattr(self, '_export_table_headers', None)
            data = getattr(self, '_export_table_data', None)
            if headers is None or data is None:
                QMessageBox.critical(self.ui, "Export Error", "No results data available for export.")
                return
            options = QFileDialog.Option.DontUseNativeDialog
            exported = False
            sidebar_locations = qfiledialog__pinned_locations()
            if fmt == 'csv':
                file_dialog = QFileDialog(self.ui, "Save Results as CSV", "results.csv", "CSV Files (*.csv)")
                file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
                file_dialog.setOptions(options)
                if sidebar_locations:
                    file_dialog.setSidebarUrls(sidebar_locations)
                if file_dialog.exec():
                    file_path = file_dialog.selectedFiles()[0]
                else:
                    file_path = ''
                if file_path:
                    with open(file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
                        writer.writerows(data)
                    QMessageBox.information(self.ui, "Export Complete", f"Results exported as CSV to:\n{file_path}")
                    exported = True
            elif fmt == 'pdf':
                file_dialog = QFileDialog(self.ui, "Save Results as PDF", "results.pdf", "PDF Files (*.pdf)")
                file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
                file_dialog.setOptions(options)
                if sidebar_locations:
                    file_dialog.setSidebarUrls(sidebar_locations)
                if file_dialog.exec():
                    file_path = file_dialog.selectedFiles()[0]
                else:
                    file_path = ''
                if file_path:
                    doc = SimpleDocTemplate(file_path, pagesize=landscape(letter))
                    elements = []
                    style = getSampleStyleSheet()["Normal"]
                    font_name = "Helvetica-Bold"
                    elements.append(Paragraph("LNVSI Tool Results", style))
                    elements.append(Spacer(1, 0.2*inch))
                    table_data = [headers] + data
                    t = Table(table_data)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.grey),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('FONTNAME', (0,0), (-1,0), font_name),
                        ('BOTTOMPADDING', (0,0), (-1,0), 12),
                        ('BACKGROUND', (0,1), (-1,-1), colors.blue),
                        ('GRID', (0,0), (-1,-1), 1, colors.black),
                    ]))
                    elements.append(t)
                    doc.build(elements)
                    QMessageBox.information(self.ui, "Export Complete", f"Results exported as PDF to:\n{file_path}")
                    exported = True
                    
            elif fmt == 'xlsx':
                file_dialog = QFileDialog(self.ui, "Save Results as XLSX", "results.xlsx", "Excel Files (*.xlsx)")
                file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
                file_dialog.setOptions(options)
                if sidebar_locations:
                    file_dialog.setSidebarUrls(sidebar_locations)
                if file_dialog.exec():
                    file_path = file_dialog.selectedFiles()[0]
                else:
                    file_path = ''
                if file_path:
                    df = pd.DataFrame(data, columns=headers)
                    df.to_excel(file_path, index=False)
                    QMessageBox.information(self.ui, "Export Complete", f"Results exported as XLSX to:\n{file_path}")
                    exported = True
            # After successful export, show only a Finish button
            if exported:
                # Remove export options from results_portal
                for i in reversed(range(self.ui.results_portal_layout.count())):
                    widget = self.ui.results_portal_layout.itemAt(i).widget()
                    if widget is not None:
                        widget.setParent(None)
                        widget.deleteLater()
                # Add Finish button
                self.finish_button = QPushButton("Finish")
                self.finish_button.setObjectName("downloadButton")
                self.finish_button.setFixedSize(250, 60)
                self.finish_button.clicked.connect(self.ui.rh.return_home_from_sucess)
                self.ui.results_portal_layout.addWidget(self.finish_button, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        except Exception as e:
            logging.error(f"Error exporting results: {e}")
            QMessageBox.critical(self.ui, "Export Error", f"An error occurred during export.\n\nError: {e}")
