# SVG to MSE Converter

A simple Python tool that converts SVG files into the **MSE (Magic Set Editor) symbol format**.

---

## Features
- Converts SVG data into **MSE-compatible** symbol files.
- Supports all path commands, basic shapes, transformations, and groupings.
- Generates a `.mse-symbol` file for use with **Magic Set Editor**.
  
**Limitations**:
- Currently **does not support lines and strokes**.
- Currently **does not support more uncommon SVG elements** such as text, patterns, and symbols.
- Transparency must be **handled by the user**; the output will include subtractive layers, but the user must apply the subtract setting themselves.

---

## Installation

1. **Download the latest release** of the [SVG to MSE Converter](https://github.com/CecilArmitais/SVGtoMSE/releases) (Windows `.exe` file).
2. **No installation necessary** – Simply **download** the `.exe` file and drag and drop 1 or more SVGs onto it.

   The first time you run, Windows may pop up a warning. If running the exe makes you uncomfortable, you can download the script and run it manually:
   - Ensure you have **Python 3.11.3** or later installed on your machine.
   - Download the `SVGtoMSE.py` and the `SVGtoMSE.bat` files.
   - Install the required dependencies by running the following command in your terminal:
     ```sh
     pip install -r requirements.txt
     ```
   - Drag and drop your SVG onto `SVGtoMSE.bat` to convert it.

---

## Usage

A. **Drag and Drop** your SVG file onto the `SVGtoMSE.exe` (Windows) to convert the SVG data to MSE format. The tool will output a `.mse-symbol` file in the same directory.

B. **Command Line Usage** (Optional):
   - Open a terminal and run:
     ```sh
     SVGtoMSE.exe <path_to_svg_file>
     ```

**Output**:
   - The program will generate a `.mse-symbol` file with coordinates formatted as required by the Magic Set Editor.
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

## Contributing

Feel free to fork this repository and submit pull requests with improvements or fixes. All contributions are welcome!