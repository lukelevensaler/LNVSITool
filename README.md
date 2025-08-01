
# LNVSI Tool (Levensaler Neogastropod Venomic Similarity Index Tool)

A PyQt6-based GUI application for venomic data analysis, spectral deconvolution, and statistical similarity indexing of neogastropod peptidomes.

## Features

- PyQt6 GUI
- Robust CSV import and JSON-based autosave
- Region-segmented spectral deconvolution (Voigt fitting, Gaussian fallback)
- Machine learning-powered similarity metrics
- FDR-corrected statistical testing
- Export results as CSV, PDF, or XLSX (to Downloads)
- Detailed logging and error handling
- Cross-platform (macOS, Linux, Windows; see build notes)

## Installation the Normal Way

- Go to [https://conowareproject.org/software/LNSVI](https://conowareproject.org/software/LNSVI) to download the installer package for your operating system

## Installation From Source

1. **Clone the repository:**

   ```sh
   git clone https://github.com/lukelevensaler/LNVSITool.git
   cd LNVSITool
   ```

2. **Create and activate a virtual environment:**

   MacOS/Linux:

   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

   Windows:

   ```sh
   python3 -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```sh
   pip install -r src/requirements.txt
   ```

4. **Compile with PyInstaller:**

    MacOS:

    - Run (with a :)

    ```sh
    pyinstaller --windowed --name "LNVSI Tool" --icon=assets/icon.png --add-data "assets:assets" src/main.py
    ```

    - The compiled `.app` bundle will be in the `dist/` directory.

    Windows:

    - Run (with a ;)

    ```sh
    pyinstaller --windowed --name "LNVSI Tool" --icon=assets/icon.png --add-data "assets;assets" src/main.py
    ```

    - The compiled `.exe` file will be located in the `dist\` directory.

    - If you see a security warning, click "More info" and "Run anyway".

    Linux:

    - You will have to first create the `.desktop` launcher in the project root:

        ```ini
        [Desktop Entry]
        Name=LNVSI Tool
        Exec=python3 /path/to/LNVSITool/src/main.py
        Icon=/path/to/LNVSITool/assets/icon.png
        Type=Application
        Categories=Science;Utility;
        ```

        Replace `/path/to/LNVSITool/` with the actual path to your project directory.

    - Then Make the `.desktop` file executable:

        ```sh
        sudo chmod +x LNVSI-Tool.desktop
        ```

    - Run (with a ;)

       ```sh
       pyinstaller --windowed --name "LNVSI Tool" --icon=assets/icon.png --add-data "assets;assets" src/main.py
       ```

    - The compiled executable will be in the `dist/` directory

    - It too must be made executable:

        ```sh
        sudo chmod +x LNVSI-Tool.desktop
        ```

## Usage Without Compilation

1. Clone the repository:

   ```sh
   git clone https://github.com/lukelevensaler/LNVSITool.git
   cd LNVSITool
   ```

2. Create and activate a virtual environment:

   ```sh
   python3 -m venv venv
   source venv/bin/activate  # or 'venv\Scripts\activate' on Windows
   ```

3. Install dependencies:

   ```sh
   pip install -r src/requirements.txt
   ```

4. Run directly from source:

   ```sh
   python src/main.py
   ```

## File Structure

- `src/` — Main application code
- `assets/` — Images, stylesheets
- `info/` — License/EULA
- `requirements.txt` — Python dependencies

## Logging

- Logs and Autosave file can be found in the LNSVI Tool Utilities Directory the application creates upon usage in your "~" directory (home directory)

## Exported Results

- All exports (CSV, PDF, XLSX) are saved to your Downloads directory by default.
- The Percentage Similarity data is saved and displayed via the main.py UI to 4 decimal places no matter what (not significant figures).
- The p-values data is, conversely, saved and displayed with 6 significant figures (not just rote decimal places).

## Full Documentation

- [https://docs.conowareproject.org](https://docs.conowareproject.org)

## License

See [info/LICENSE.md](info/LICENSE.md)

---

Created by Luke Levensaler, 2025
