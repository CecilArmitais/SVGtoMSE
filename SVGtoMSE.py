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

def parse_path_commands(d_attr):
    """
    Parses an SVG path 'd' attribute string into a list of command dictionaries.
    
    Each dictionary has:
      - "cmd": the command letter (e.g., "M", "C", "l", etc.)
      - "args": a list of numeric arguments for that command.
      
    This regex splits on any command letter.
    """
    # This pattern captures a command letter followed by everything until the next command letter.
    command_pattern = re.compile(r'([MmZzLlHhVvCcSsQqTtAa])([^MmZzLlHhVvCcSsQqTtAa]*)')
    commands = []
    for match in command_pattern.findall(d_attr):
        cmd = match[0]
        args_str = match[1].strip()
        if args_str:
            # This regex captures numbers, including decimal and scientific notation.
            args = list(map(float, re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', args_str)))
        else:
            args = []
        commands.append({"cmd": cmd, "args": args})
    return commands

def extract_subpaths(svg_file):
    """
    Extracts all paths from an SVG file and splits them into subpaths.
    
    A subpath is defined as a sequence of path commands starting with a moveto (M/m) 
    command and continuing until the next moveto.
    
    Returns a list of subpaths, where each subpath is a list of command dictionaries.
    Also prints an error message if there are unsupported elements (ignoring the <svg> root).
    """
    try:
        tree = ET.parse(svg_file)
        root = tree.getroot()

        # Find unsupported elements (ignoring <svg>)
        unsupported_elements = [
            elem.tag for elem in root.iter()
            if elem.tag not in ("{http://www.w3.org/2000/svg}svg", "{http://www.w3.org/2000/svg}path")
        ]
        if unsupported_elements:
            print(f"ERROR - Unsupported elements: {len(unsupported_elements)} objects failed to convert.")

        # Find all <path> elements in the SVG
        path_elements = root.findall(".//{http://www.w3.org/2000/svg}path")
        if not path_elements:
            print("No <path> elements found in SVG.")
            return None

        all_subpaths = []
        for path_elem in path_elements:
            d_attr = path_elem.get("d")
            if not d_attr:
                print("Skipping a <path> without 'd' attribute.")
                continue
            # Parse all commands from the d attribute.
            commands = parse_path_commands(d_attr)
            # Split commands into subpaths
            subpaths = []
            current_subpath = []
            for command in commands:
                if command["cmd"] in ["M", "m"]:
                    # If current_subpath is not empty, a new moveto indicates a new subpath.
                    if current_subpath:
                        subpaths.append(current_subpath)
                        current_subpath = []
                current_subpath.append(command)
            if current_subpath:
                subpaths.append(current_subpath)
            all_subpaths.extend(subpaths)
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
    Only supports M/m and C/c commands; others are skipped.
    """
    relativehandle = 1  # Determines the ratio from MSE to SVG handle strength.

    viewbox = get_viewbox(svg_file)
    if not viewbox:
        return "Error: No valid viewBox found."

    subpaths = extract_subpaths(svg_file)
    if not subpaths:
        return "Error: No valid subpaths found."

    mse_output = "mse_version: 0.3.5"

    # Process each subpath (each is a list of command dictionaries)
    for subpath in subpaths:
        # Build segments from commands in the subpath.
        segments = []
        current_point = None
        for command in subpath:
            cmd = command["cmd"]
            args = command["args"]
            if cmd in ["M", "m"]:
                # Set starting point.
                if cmd == "M":
                    current_point = (args[0], args[1])
                else:  # 'm' is relative
                    if current_point is None:
                        current_point = (args[0], args[1])
                    else:
                        current_point = (current_point[0] + args[0], current_point[1] + args[1])
            elif cmd in ["C", "c"]:
                if current_point is None:
                    continue  # Skip if no starting point.
                if cmd == "C":
                    c1 = (args[0], args[1])
                    c2 = (args[2], args[3])
                    end = (args[4], args[5])
                else:  # 'c' is relative
                    c1 = (current_point[0] + args[0], current_point[1] + args[1])
                    c2 = (current_point[0] + args[2], current_point[1] + args[3])
                    end = (current_point[0] + args[4], current_point[1] + args[5])
                segment = (current_point[0], current_point[1],
                           c1[0], c1[1],
                           c2[0], c2[1],
                           end[0], end[1])
                segments.append(segment)
                current_point = end
            else:
                # Skip any unsupported command.
                continue

        # If no segments were built, skip this subpath.
        if not segments:
            continue

        mse_output += """
part:
	type: shape
	name: polygon
	combine: overlap
"""
        mse_points = []

        # For the first point in the subpath, use the last segment's second control point (c2)
        last_seg = segments[-1]
        last_c2_x, last_c2_y = normalize_coordinates(last_seg[4], last_seg[5], viewbox)

        # Process each segment
        for i, seg in enumerate(segments):
            start_x, start_y, c1_x, c1_y, c2_x, c2_y, end_x, end_y = seg

            norm_start_x, norm_start_y = normalize_coordinates(start_x, start_y, viewbox)
            norm_c1_x, norm_c1_y = normalize_coordinates(c1_x, c1_y, viewbox)
            norm_c2_x, norm_c2_y = normalize_coordinates(c2_x, c2_y, viewbox)
            norm_end_x, norm_end_y = normalize_coordinates(end_x, end_y, viewbox)

            # For the first segment, use the last segment's c2 as handle_before.
            if i == 0:
                handle_before_x = (last_c2_x - norm_start_x) * relativehandle
                handle_before_y = (last_c2_y - norm_start_y) * relativehandle
            else:
                prev_seg = segments[i-1]
                norm_prev_c2_x, norm_prev_c2_y = normalize_coordinates(prev_seg[4], prev_seg[5], viewbox)
                handle_before_x = (norm_prev_c2_x - norm_start_x) * relativehandle
                handle_before_y = (norm_prev_c2_y - norm_start_y) * relativehandle

            # The handle_after for the starting point is the offset from start to c1.
            handle_after_x = (norm_c1_x - norm_start_x) * relativehandle
            handle_after_y = (norm_c1_y - norm_start_y) * relativehandle

            mse_points.append(f"""	point:
		position: ({norm_start_x:.6f},{norm_start_y:.6f})
		lock: free
		line_after: curve
		handle_before: ({handle_before_x:.6f},{handle_before_y:.6f})
		handle_after: ({handle_after_x:.6f},{handle_after_y:.6f})""")
            
        mse_output += "\n" + "\n".join(mse_points)
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