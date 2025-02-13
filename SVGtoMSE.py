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
import math

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



#Refactored  parts of extract_subpaths():

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



#Refactored parts of build_segments():

def arc_segment(cx, cy, rx, ry, phi, t1, t2):
	"""
	Approximates one segment of an elliptical arc (from angle t1 to t2) 
	with a cubic Bézier curve.
	
	Returns a tuple (c1, c2, endpoint) where all points are in the original coordinate system.
	"""
	dt = t2 - t1
	# Compute alpha as in the standard formula.
	alpha = math.sin(dt) * (math.sqrt(4 + 3 * (math.tan(dt/2)**2)) - 1) / 3.0

	# Starting point of the segment on the ellipse in the ellipse's local coordinate system.
	x1 = rx * math.cos(t1)
	y1 = ry * math.sin(t1)
	# End point of the segment in the ellipse's local coordinate system.
	x2 = rx * math.cos(t2)
	y2 = ry * math.sin(t2)

	# First control point (local)
	c1_local = (x1 - alpha * rx * math.sin(t1),
				y1 + alpha * ry * math.cos(t1))
	# Second control point (local)
	c2_local = (x2 + alpha * rx * math.sin(t2),
				y2 - alpha * ry * math.cos(t2))

	# Define a transformation that rotates by phi and translates by (cx, cy)
	cos_phi = math.cos(phi)
	sin_phi = math.sin(phi)
	def transform(pt):
		x, y = pt
		xp = cos_phi * x - sin_phi * y + cx
		yp = sin_phi * x + cos_phi * y + cy
		return (xp, yp)

	c1 = transform(c1_local)
	c2 = transform(c2_local)
	end = transform((x2, y2))
	return c1, c2, end

def arc_to_cubic(x1, y1, rx, ry, phi_deg, large_arc_flag, sweep_flag, x2, y2):
	"""
	Converts an elliptical arc (from (x1,y1) to (x2,y2)) with parameters rx, ry, 
	x-axis rotation (in degrees), large_arc_flag, and sweep_flag into one or more cubic Bézier segments.
	
	Returns a list of segments, each of the form: ("C", start, c1, c2, end)
	"""
	# Convert phi from degrees to radians.
	phi = math.radians(phi_deg)
	
	# Step 1: Compute (x1', y1') – the coordinates of the start point in the transformed coordinate system.
	dx = (x1 - x2) / 2.0
	dy = (y1 - y2) / 2.0
	cos_phi = math.cos(phi)
	sin_phi = math.sin(phi)
	x1p = cos_phi * dx + sin_phi * dy
	y1p = -sin_phi * dx + cos_phi * dy

	# Ensure radii are positive.
	rx = abs(rx)
	ry = abs(ry)

	# Step 2: Correct radii if they are too small.
	lambda_val = (x1p**2)/(rx**2) + (y1p**2)/(ry**2)
	if lambda_val > 1:
		factor = math.sqrt(lambda_val)
		rx *= factor
		ry *= factor

	# Step 3: Compute center coordinates in the transformed system.
	sign = -1 if large_arc_flag == sweep_flag else 1
	numerator = rx**2 * ry**2 - rx**2 * y1p**2 - ry**2 * x1p**2
	# Avoid negative under the square root due to floating point errors.
	denom = rx**2 * y1p**2 + ry**2 * x1p**2
	factor = sign * math.sqrt(max(0, numerator/denom)) if denom != 0 else 0
	cxp = factor * (rx * y1p) / ry
	cyp = factor * (-ry * x1p) / rx

	# Step 4: Compute the center in the original coordinate system.
	cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2)/2.0
	cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2)/2.0

	# Step 5: Compute the start and delta angles.
	def vector_angle(u, v):
		dot = u[0]*v[0] + u[1]*v[1]
		len_u = math.hypot(u[0], u[1])
		len_v = math.hypot(v[0], v[1])
		# Clamp dot/(len_u*len_v) between -1 and 1 to avoid domain errors.
		angle = math.acos(max(min(dot/(len_u*len_v), 1), -1))
		if u[0]*v[1] - u[1]*v[0] < 0:
			return -angle
		return angle

	# Compute angle between (1,0) and ( (x1p-cxp)/rx, (y1p-cyp)/ry ).
	v1 = ((x1p - cxp)/rx, (y1p - cyp)/ry)
	v2 = ((-x1p - cxp)/rx, (-y1p - cyp)/ry)
	theta1 = vector_angle((1,0), v1)
	delta_theta = vector_angle(v1, v2)

	if not sweep_flag and delta_theta > 0:
		delta_theta -= 2 * math.pi
	elif sweep_flag and delta_theta < 0:
		delta_theta += 2 * math.pi

	# Step 6: Determine the number of segments (each segment at most pi/2).
	num_segments = int(math.ceil(abs(delta_theta) / (math.pi/2)))
	seg_list = []
	delta = delta_theta / num_segments
	t = theta1
	current = (x1, y1)
	for i in range(num_segments):
		c1, c2, ep = arc_segment(cx, cy, rx, ry, phi, t, t+delta)
		seg_list.append(("C", current, c1, c2, ep))
		current = ep
		t += delta
	return seg_list



#Refactored parts of convert_to_mse():

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

def build_segments(subpath):
	"""
	Build a list of segments from a subpath.
	Each segment is:
	  - For curves: ("C", start, c1, c2, end)
	  - For lines:   ("L", start, end)
	Supports:
	  - M/m (moveto)
	  - C/c (cubic Bézier)
	  - S/s (smooth cubic Bézier)
	  - Q/q (quadratic Bézier, converted to cubic)
	  - T/t (smooth quadratic, converted to cubic)
	  - A/a (arc, converted to cubic)
	  - L/l (lineto)
	  - H/h (horizontal lineto)
	  - V/v (vertical lineto)
	"""
	segments = []
	current_point = None
	start_point = None  # Store the first point of a subpath
	last_quad_ctrl = None  # For smooth quadratic
	prev_command = None  # For S/s and T/t reflection

	for command in subpath:
		cmd = command["cmd"]
		args = command["args"]

		if cmd in ["M", "m"]:
			if cmd == "M":
				current_point = (args[0], args[1])
			else:
				if current_point is None:
					current_point = (args[0], args[1])
				else:
					current_point = (current_point[0] + args[0], current_point[1] + args[1])
			start_point = current_point  # Store subpath start
			last_quad_ctrl = None
			prev_command = cmd

		elif cmd in ["C", "c"]:
			if current_point is None:
				continue
			if cmd == "C":
				c1 = (args[0], args[1])
				c2 = (args[2], args[3])
				end = (args[4], args[5])
			else:  # 'c' is relative
				c1 = (current_point[0] + args[0], current_point[1] + args[1])
				c2 = (current_point[0] + args[2], current_point[1] + args[3])
				end = (current_point[0] + args[4], current_point[1] + args[5])
			segments.append(("C", current_point, c1, c2, end))
			current_point = end

		elif cmd in ["L", "l"]:
			if current_point is None:
				continue
			if cmd == "L":
				# For L, we expect pairs of coordinates.
				# Process them in pairs.
				for i in range(0, len(args), 2):
					end = (args[i], args[i+1])
					segments.append(("L", current_point, end))
					current_point = end
			else:  # 'l' is relative
				for x_delta, y_delta in zip(args[::2], args[1::2]):
					end = (current_point[0] + x_delta, current_point[1] + y_delta)
					segments.append(("L", current_point, end))
					current_point = end

		elif cmd in ["S", "s"]:
			if current_point is None:
				continue
			# Compute the first control point as a reflection if the previous command was C or S.
			if prev_command in ["C", "S"] and segments:
				last_seg = segments[-1]
				if last_seg[0] == "C":
					last_c2 = last_seg[3]
				else:
					last_c2 = current_point  # Fallback.
				computed_c1 = (2 * current_point[0] - last_c2[0],
							   2 * current_point[1] - last_c2[1])
			else:
				computed_c1 = current_point

			if cmd == "S":
				c2 = (args[0], args[1])
				end = (args[2], args[3])
			else:  # relative s
				c2 = (current_point[0] + args[0], current_point[1] + args[1])
				end = (current_point[0] + args[2], current_point[1] + args[3])
			segments.append(("C", current_point, computed_c1, c2, end))
			current_point = end
			prev_command = cmd

		elif cmd in ["Q", "q"]:
			if current_point is None:
				continue
			# Quadratic curve: parameters: control point, endpoint.
			if cmd == "Q":
				quad_ctrl = (args[0], args[1])
				end = (args[2], args[3])
			else:  # relative quadratic
				quad_ctrl = (current_point[0] + args[0], current_point[1] + args[1])
				end = (current_point[0] + args[2], current_point[1] + args[3])
			# Convert quadratic to cubic.
			c1 = (current_point[0] + (2/3) * (quad_ctrl[0] - current_point[0]),
				  current_point[1] + (2/3) * (quad_ctrl[1] - current_point[1]))
			c2 = (end[0] + (2/3) * (quad_ctrl[0] - end[0]),
				  end[1] + (2/3) * (quad_ctrl[1] - end[1]))
			segments.append(("C", current_point, c1, c2, end))
			current_point = end
			last_quad_ctrl = quad_ctrl  # Store for T/t commands.
			prev_command = cmd
			
		elif cmd in ["T", "t"]:
			if current_point is None:
				continue
			# Smooth quadratic: if previous was Q or T, reflect its control point.
			if last_quad_ctrl is not None:
				quad_ctrl = (2 * current_point[0] - last_quad_ctrl[0],
							 2 * current_point[1] - last_quad_ctrl[1])
			else:
				quad_ctrl = current_point
			if cmd == "T":
				end = (args[0], args[1])
			else:  # relative
				end = (current_point[0] + args[0], current_point[1] + args[1])
			# Convert quadratic to cubic.
			c1 = (current_point[0] + (2/3) * (quad_ctrl[0] - current_point[0]),
				  current_point[1] + (2/3) * (quad_ctrl[1] - current_point[1]))
			c2 = (end[0] + (2/3) * (quad_ctrl[0] - end[0]),
				  end[1] + (2/3) * (quad_ctrl[1] - end[1]))
			segments.append(("C", current_point, c1, c2, end))
			current_point = end
			last_quad_ctrl = quad_ctrl
			prev_command = cmd

		elif cmd in ["H", "h"]:
			# Horizontal lineto: only x value(s); y remains the same.
			if current_point is None:
				continue
			if cmd == "H":
				for x_val in args:
					end = (x_val, current_point[1])
					segments.append(("L", current_point, end))
					current_point = end
			else:  # 'h' is relative
				for x_delta in args:
					end = (current_point[0] + x_delta, current_point[1])
					segments.append(("L", current_point, end))
					current_point = end

		elif cmd in ["V", "v"]:
			# Vertical lineto: only y value(s); x remains the same.
			if current_point is None:
				continue
			if cmd == "V":
				for y_val in args:
					end = (current_point[0], y_val)
					segments.append(("L", current_point, end))
					current_point = end
			else:  # 'v' is relative
				for y_delta in args:
					end = (current_point[0], current_point[1] + y_delta)
					segments.append(("L", current_point, end))
					current_point = end

		elif cmd in ["A", "a"]:
			# Arc command: parameters: rx, ry, phi, large_arc_flag, sweep_flag, x, y
			if current_point is None:
				continue
			if cmd == "A":
				rx, ry, phi_deg, laf, sf, x, y = args
				end = (x, y)
			else:
				rx, ry, phi_deg, laf, sf, dx, dy = args
				end = (current_point[0] + dx, current_point[1] + dy)
			# Convert arc to one or more cubic Bézier segments.
			arc_segs = arc_to_cubic(current_point[0], current_point[1],
									rx, ry, phi_deg, int(laf), int(sf),
									end[0], end[1])
			for seg in arc_segs:
				segments.append(seg)
			current_point = end
			last_quad_ctrl = None
			prev_command = cmd

		elif cmd in ["Z", "z"]:
			if current_point is not None and start_point is not None and current_point != start_point:
				segments.append(("L", current_point, start_point))
				current_point = start_point  # Ensure closure
			last_quad_ctrl = None
			prev_command = cmd

		else:
			# Skip any unsupported command.
			continue
	return segments

def format_segment(seg, index, segments, viewbox):
    """
    Returns the formatted MSE point for a segment.
    - Curves (C) output: 'line_after: curve' and may have both handles.
    - Lines (L) output: 'line_after: line' and may have only handle_before.
    """
    segment_type = seg[0]
    start = seg[1]  # Common start point
    outputtext = "	point:"

    norm_start = normalize_coordinates(start[0], start[1], viewbox)

    # Determine the previous segment (the one before the current segment in MSE format)
    prev_seg = segments[len(segments) - 1] if index == 0 else segments[index - 1]

    # Handle before calculation: for the first point, use last_c2 (the final c2 of the previous segment)
    if prev_seg[0] == "C":
        norm_prev_c2 = normalize_coordinates(prev_seg[3][0], prev_seg[3][1], viewbox)
    else:
        norm_prev_c2 = normalize_coordinates(prev_seg[1][0], prev_seg[1][1], viewbox)
    
    handle_before = (norm_prev_c2[0] - norm_start[0], norm_prev_c2[1] - norm_start[1])

    # Case for curves (C) and lines (L)
    if segment_type == "C":
        # Unpack control points for curves
        c1 = seg[2]
        norm_c1 = normalize_coordinates(c1[0], c1[1], viewbox)

        # Handle after calculation for curves
        handle_after = (norm_c1[0] - norm_start[0], norm_c1[1] - norm_start[1])

        outputtext += f"""
		position: ({norm_start[0]:.6f},{norm_start[1]:.6f})
		lock: free
		line_after: curve"""

        # If coming from a curve, output both handles
        if prev_seg[0] == "C":
            outputtext += f"""
		handle_before: ({handle_before[0]:.6f},{handle_before[1]:.6f})
		handle_after: ({handle_after[0]:.6f},{handle_after[1]:.6f})"""
        # If coming from a line, output only handle_after
        elif prev_seg[0] == "L":
            outputtext += f"""
		handle_after: ({handle_after[0]:.6f},{handle_after[1]:.6f})"""

    elif segment_type == "L":
        # Unpack line endpoint
        end = seg[2]
        outputtext += f"""
		position: ({norm_start[0]:.6f},{norm_start[1]:.6f})
		lock: free
		line_after: line"""

        # If coming from a curve, output handle_before
        if prev_seg[0] == "C":
            outputtext += f"""
		handle_before: ({handle_before[0]:.6f},{handle_before[1]:.6f})"""

    else:
        print("Unknown segment type detected.")

    return outputtext


	
def convert_to_mse(svg_file):
	"""
	Converts an SVG file's paths into MSE format, handling multiple subpaths.
	Supports M/m, C/c, and L/l commands; others are skipped.
	"""
	viewbox = get_viewbox(svg_file)
	if not viewbox:
		return "Error: No valid viewBox found."

	subpaths = extract_subpaths(svg_file)
	if not subpaths:
		return "Error: No valid subpaths found."

	mse_output = "mse_version: 0.3.5"

	for subpath in subpaths:
		segments = build_segments(subpath)
		if not segments:
			continue

		mse_output += """
part:
	type: shape
	name: polygon
	combine: overlap
"""
		mse_points = []

		for i, seg in enumerate(segments):
			mse_points.append(format_segment(seg, i, segments, viewbox))
			
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