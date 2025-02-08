# SVG to MSE Converter

A simple Python tool that converts SVG path data into the **MSE (Magic Set Editor) symbol format**.

---

## Features
- Converts SVG paths into **MSE-compatible** symbol files.
- Supports some basic path commands: `M` (Move), `C` (Curve).
- Generates a `.mse-symbol` file for use with **Magic Set Editor**.
  
**Limitations**:
- Currently **does not support `L` (Line) commands** – Straight line segments are skipped, and only curves are processed.
- Currently **does not support other SVG elements** such as `<circle>`, `<rect>`, `<polygon>`, etc. Only `<path>` elements with valid `d` attributes are converted.

---

## Installation

1. **Download the latest release** of the [SVG to MSE Converter](https://github.com/CecilArmitais/SVGtoMSE/releases) (Windows `.exe` file).
2. **No installation necessary** – Simply **download** the `.exe` file and drag and drop 1 or more SVGs onto it.

   The first time you run, windows may pop up a warning. If running the exe makes you uncomfortable, you can download the script and run it manually:
   - Ensure you have **Python 3.11.3** installed on your machine.
   - Download the `SVGtoMSE.py` and the `SVGtoMSE.bat` files.
   - Drag and drop your SVG onto `SVGtoMSE.bat`.

---

## Usage

A. **Drag and Drop** your SVG file onto the `SVGtoMSE.exe` (Windows) to convert the SVG path data to MSE format. The tool will output a `.mse-symbol` file in the same directory.

B. **Command Line Usage** (Optional):
   - Open a terminal and run:
     ```sh
     SVGtoMSE.exe <path_to_svg_file>
     ```

**Supported Input Format**:
   - The SVG file should contain `<path>` elements with valid `d` attributes.
   - Example input:
     ```xml
     <path d="M10,10 C20,5 30,15 40,20"/>
     ```

**Output**:
   - The program will generate a `.mse-symbol` file with relative coordinates formatted as required by the Magic Set Editor.
   - Example output (partial):
     ```
	 mse_version: 0.3.5
     part:
       type: shape
       name: polygon
       combine: overlap
     point:
       position: (0.0, 0.0)
       lock: free
       line_after: curve
       handle_before: (0.1, 0.1)
       handle_after: (0.2, 0.2)
     ```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Troubleshooting

- **ERROR - Unsupported elements**: This message appears when the SVG contains elements other than `<path>` (e.g., `<circle>`, `<rect>`). These elements are skipped, but any other paths within the file are still converted successfully.

---

## Contributing

Feel free to fork this repository and submit pull requests with improvements or fixes. All contributions are welcome!

