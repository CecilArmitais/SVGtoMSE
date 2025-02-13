# SVGtoMSE.py

# MIT License
# 
# Copyright (c) 2025 Joseph Duval
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import os
import re
import xml.etree.ElementTree as ET

def get_viewbox(svg_file):
    """
    Extracts the viewBox attribute from an SVG file.
    
    Parameters:
    - svg_file: Path to the SVG file.
    
    Returns:
    - A tuple (min_x, min_y, width, height) if viewBox exists.
    - None if no viewBox is found.
    """
    try:
        tree = ET.parse(svg_file)
        root = tree.getroot()

        # The 'viewBox' attribute is in the <svg> tag
        viewbox_str = root.get("viewBox")

        if viewbox_str:
            # Split the viewBox string into four float values
            min_x, min_y, width, height = map(float, viewbox_str.split())
            return min_x, min_y, width, height
        else:
            print("No viewBox attribute found in SVG.")
            return None

    except ET.ParseError as e:
        print(f"Error parsing SVG file: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None





def normalize_coordinates(x, y, viewbox):
    """
    Normalize SVG coordinates to the MSE format (0 to 1 scale).
    
    Parameters:
    - x, y: The absolute coordinates from the SVG.
    - viewbox: A tuple (min_x, min_y, width, height) from the SVG viewBox.
    
    Returns:
    - (norm_x, norm_y): The normalized coordinates in the range [0,1].
    """
    min_x, min_y, width, height = viewbox
    
    # Ensure width and height are not zero to avoid division errors
    if width == 0 or height == 0:
        raise ValueError("Invalid viewBox dimensions: width and height must be nonzero.")

    height_adjustment = 0
    width_adjustment = 0

    if width > height:
        norm_ratio = width
        height_adjustment = (width - height) / 2
    else:
        norm_ratio = height
        width_adjustment = (height - width) / 2

    # Normalize x and y
    norm_x = (x - min_x + width_adjustment) / norm_ratio
    norm_y = (y - min_y + height_adjustment) / norm_ratio

    return norm_x, norm_y

def extract_subpaths(svg_file):
    """
    Extracts all paths from an SVG file, splitting each path into subpaths at 'M' commands.
    Returns a list of lists, where each inner list represents a subpath.
    """
    try:
        tree = ET.parse(svg_file)
        root = tree.getroot()

        # Find all <path> elements in the SVG
        path_elements = root.findall(".//{http://www.w3.org/2000/svg}path")
        
        unsupported_elements = [
            elem.tag for elem in root.iter()
            if elem.tag != "{http://www.w3.org/2000/svg}path" and elem.tag != "{http://www.w3.org/2000/svg}svg"
        ]
        
        if unsupported_elements:
            print(f"ERROR - Unsupported elements: {len(unsupported_elements)} objects failed to convert.")

        if not path_elements:
            print("No <path> elements found in SVG.")
            return None

        all_subpaths = []
        path_pattern = re.compile(r"M([\d.-]+),([\d.-]+)|C([\d.-]+),([\d.-]+) ([\d.-]+),([\d.-]+) ([\d.-]+),([\d.-]+)")

        for path_element in path_elements:
            d_attr = path_element.get("d")
            if not d_attr:
                print("Skipping a <path> without 'd' attribute.")
                continue

            matches = path_pattern.findall(d_attr)

            current_subpath = []
            current_x, current_y = None, None
            path_subpaths = []

            for match in matches:
                if match[0]:  # 'M' command found, starting a new subpath
                    if current_subpath:
                        path_subpaths.append(current_subpath)  # Store previous subpath
                    current_subpath = []  # Start a new subpath
                    current_x, current_y = float(match[0]), float(match[1])
                else:  # 'C' command found
                    c1_x, c1_y = float(match[2]), float(match[3])
                    c2_x, c2_y = float(match[4]), float(match[5])
                    end_x, end_y = float(match[6]), float(match[7])

                    if current_x is None or current_y is None:
                        print("Error: 'C' command found before 'M' command.")
                        return None

                    current_subpath.append((current_x, current_y, c1_x, c1_y, c2_x, c2_y, end_x, end_y))
                    current_x, current_y = end_x, end_y  # Update position

            if current_subpath:
                path_subpaths.append(current_subpath)  # Add last subpath

            all_subpaths.extend(path_subpaths)

        return all_subpaths

    except ET.ParseError as e:
        print(f"Error parsing SVG file: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    
def convert_to_mse(svg_file):
    """
    Converts an SVG file's paths into MSE format, handling multiple subpaths.
    """
    relativehandle = 1 # Determines the ratio from MSE to SVG handle strength.

    viewbox = get_viewbox(svg_file)
    if not viewbox:
        return "Error: No valid viewBox found."

    subpaths = extract_subpaths(svg_file)
    if not subpaths:
        return "Error: No valid subpaths found."

    mse_output = "mse_version: 0.3.5"

    for subpath in subpaths:
        mse_output += """
part:
	type: shape
	name: polygon
	combine: overlap
"""

        mse_points = []

        # Get the last control point of the subpath for the first point's handle_before
        last_c2_x, last_c2_y = normalize_coordinates(subpath[-1][4], subpath[-1][5], viewbox)

        for i, (start_x, start_y, c1_x, c1_y, c2_x, c2_y, end_x, end_y) in enumerate(subpath):
            norm_start_x, norm_start_y = normalize_coordinates(start_x, start_y, viewbox)
            norm_c1_x, norm_c1_y = normalize_coordinates(c1_x, c1_y, viewbox)
            norm_c2_x, norm_c2_y = normalize_coordinates(c2_x, c2_y, viewbox)
            norm_end_x, norm_end_y = normalize_coordinates(end_x, end_y, viewbox)

            if i == 0:
                # First point uses the last c2 control point relative to the start position
                handle_before_x = (last_c2_x - norm_start_x) * relativehandle
                handle_before_y = (last_c2_y - norm_start_y) * relativehandle
            else:
                _, _, prev_c1_x, prev_c1_y, prev_c2_x, prev_c2_y, prev_end_x, prev_end_y = subpath[i - 1]
                norm_prev_c2_x, norm_prev_c2_y = normalize_coordinates(prev_c2_x, prev_c2_y, viewbox)
                handle_before_x = (norm_prev_c2_x - norm_start_x) * relativehandle
                handle_before_y = (norm_prev_c2_y - norm_start_y) * relativehandle

            # Handles relative to the point
            handle_after_x = (norm_c1_x - norm_start_x) * relativehandle
            handle_after_y = (norm_c1_y - norm_start_y) * relativehandle

            mse_points.append(f"""	point:
		position: ({norm_start_x:.6f},{norm_start_y:.6f})
		lock: free
		line_after: curve
		handle_before: ({handle_before_x:.6f},{handle_before_y:.6f})
		handle_after: ({handle_after_x:.6f},{handle_after_y:.6f})""")

        mse_output += "\n" + "\n".join(mse_points)  # Append points without extra line breaks
    return mse_output





def process_file(input_file):
    # Ensure the file exists and is an SVG file
    if not os.path.isfile(input_file):
        print(f"File not found: {input_file}")
        return

    if not input_file.lower().endswith('.svg'):
        print(f"Skipping non-SVG file: {input_file}")
        return

    # Create the output filename by replacing .svg with .mse-symbol
    output_file = os.path.splitext(input_file)[0] + ".mse-symbol"

    MSE_OUTPUT = convert_to_mse(input_file)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(MSE_OUTPUT)
        print(f"Created: {output_file}")
    except Exception as e:
        print(f"Error writing {output_file}: {e}")





def main():
    # Ensure at least one file was provided
    if len(sys.argv) < 2:
        print("Usage: Drag and drop an SVG file onto this script to create an empty .mse-symbol file.")
        sys.exit(1)

    # Process each file provided as an argument
    for file_path in sys.argv[1:]:
        process_file(file_path)

if __name__ == "__main__":
    main()