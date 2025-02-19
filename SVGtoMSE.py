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
import numpy as np
import math

#Global Counters
path_count = 1
rect_count = 1
circle_count = 1
ellipse_count = 1
polygon_count = 1

def get_viewbox(svg_file):
	"""
	Extracts the viewBox attribute from an SVG file, with fallbacks.
	
	Parameters:
	- svg_file: Path to the SVG file.
	
	Returns:
	- A tuple (min_x, min_y, width, height) if viewBox exists or is inferred.
	"""
	try:
		tree = ET.parse(svg_file)
		root = tree.getroot()

		# Check for viewBox attribute
		viewbox_str = root.get("viewBox")
		if viewbox_str:
			try:
				min_x, min_y, width, height = map(float, viewbox_str.split())
				return min_x, min_y, width, height
			except ValueError:
				print("ERROR - Malformed viewBox attribute.")
				return None

		# Fallback: Try using width and height attributes
		width = root.get("width")
		height = root.get("height")

		if width and height:
			try:
				width = float(width)
				height = float(height)
				return 0.0, 0.0, width, height
			except ValueError:
				print("ERROR - Malformed width/height attributes.")
				return None

		# Final fallback if all else fails
		print("WARNING - No viewBox, width, or height found. Using default (0,0,100,100).")
		return 0.0, 0.0, 100.0, 100.0

	except ET.ParseError as e:
		print(f"ERROR - Failed to parse SVG file: {e}")
		return None
	except Exception as e:
		print(f"ERROR - Unexpected issue: {e}")
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



#Refactored  parts of extract_elements():

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

def arc_to_cubic(rx, ry, x_axis_rotation, large_arc_flag, sweep_flag, end_x, end_y, start_x, start_y):
	def radians(degrees):
		return degrees * (math.pi / 180)
	
	def rotate_point(x, y, angle):
		"""Rotate a point by a given angle (in radians)."""
		cos_a = math.cos(angle)
		sin_a = math.sin(angle)
		return x * cos_a - y * sin_a, x * sin_a + y * cos_a

	# Step 1: Handle radii issues
	rx, ry = abs(rx), abs(ry)
	if rx == 0 or ry == 0:
		return [{"cmd": "L", "args": [end_x, end_y]}]  # Treat as a straight line

	# Step 2: Convert to center parameterization
	dx = (start_x - end_x) / 2
	dy = (start_y - end_y) / 2
	x_rot = radians(x_axis_rotation)
	x1p, y1p = rotate_point(dx, dy, -x_rot)

	# Ensure radii are large enough
	radius_check = (x1p**2) / (rx**2) + (y1p**2) / (ry**2)
	if radius_check > 1:
		scale = math.sqrt(radius_check)
		rx *= scale
		ry *= scale

	# Step 3: Find the center of the ellipse
	sign = -1 if large_arc_flag == sweep_flag else 1
	cxp = sign * math.sqrt(abs((rx**2 * ry**2 - rx**2 * y1p**2 - ry**2 * x1p**2) / (rx**2 * y1p**2 + ry**2 * x1p**2))) * (rx * y1p / ry)
	cyp = sign * math.sqrt(abs((rx**2 * ry**2 - rx**2 * y1p**2 - ry**2 * x1p**2) / (rx**2 * y1p**2 + ry**2 * x1p**2))) * (-ry * x1p / rx)
	
	cx, cy = rotate_point(cxp, cyp, x_rot)
	cx += (start_x + end_x) / 2
	cy += (start_y + end_y) / 2

	# Step 4: Compute angles
	def angle_between(v1, v2):
		"""Compute angle between two vectors."""
		dot = v1[0] * v2[0] + v1[1] * v2[1]
		mag = math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2)
		return math.acos(max(-1, min(1, dot / mag))) * (-1 if (v1[0] * v2[1] - v1[1] * v2[0]) < 0 else 1)

	start_angle = angle_between([1, 0], [(x1p - cxp) / rx, (y1p - cyp) / ry])
	delta_angle = angle_between([(x1p - cxp) / rx, (y1p - cyp) / ry], [(end_x - cx) / rx, (end_y - cy) / ry])
	
	if not sweep_flag and delta_angle > 0:
		delta_angle -= 2 * math.pi
	elif sweep_flag and delta_angle < 0:
		delta_angle += 2 * math.pi

	num_segments = max(1, int(math.ceil(abs(delta_angle) / (math.pi / 2))))  # Split into 90-degree segments
	segment_angle = delta_angle / num_segments

	# Step 5: Convert each arc segment to a cubic Bézier curve
	arc_commands = []
	for i in range(num_segments):
		angle1 = start_angle + i * segment_angle
		angle2 = start_angle + (i + 1) * segment_angle
		cos_a1, sin_a1 = math.cos(angle1), math.sin(angle1)
		cos_a2, sin_a2 = math.cos(angle2), math.sin(angle2)

		x1 = cx + rx * cos_a1
		y1 = cy + ry * sin_a1
		x2 = cx + rx * cos_a2
		y2 = cy + ry * sin_a2

		alpha = math.tan((angle2 - angle1) / 4) * 4 / 3
		control1_x = x1 - alpha * rx * sin_a1
		control1_y = y1 + alpha * ry * cos_a1
		control2_x = x2 + alpha * rx * sin_a2
		control2_y = y2 - alpha * ry * cos_a2

		arc_commands.append({"cmd": "C", "args": [control1_x, control1_y, control2_x, control2_y, x2, y2]})

	return arc_commands

def parse_transformations(transform_str):
	"""
	Parses the 'transform' attribute from an SVG element and returns a list of transformations.
	Each transformation is represented as a dictionary with the type and arguments.
	"""
	transform_list = []
	transform_regex = re.findall(r"(\w+)\(([^)]+)\)", transform_str)

	for t_type, values in transform_regex:
		args = list(map(float, values.replace(",", " ").split()))
		transform_list.append({"type": t_type, "args": args})

	return transform_list

def calculate_winding_order(commands):
	"""
	Calculate the winding order of a path based on its commands.
	Returns a positive value if the path is clockwise, and negative if counterclockwise.
	"""
	area = 0
	n = len(commands)
	last_x, last_y = 0, 0  # Initialize the starting point of the path
	
	for i in range(n):
		cmd = commands[i]["cmd"]
		args = commands[i]["args"]

		if cmd in ["M", "L", "T", "S", "C", "Q", "A"]:
			# Absolute commands: take the last two args as coordinates
			x1, y1 = args[-2], args[-1]
		elif cmd in ["m", "l", "t", "s", "c", "q", "a"]:
			# Relative commands: add to the last coordinates (based on previous position)
			x1, y1 = last_x + args[-2], last_y + args[-1]
		elif cmd in ["H"]:
			# Horizontal line, only needs the x-coordinate, use the previous y
			x1, y1 = args[0], last_y
		elif cmd in ["V"]:
			# Vertical line, only needs the y-coordinate, use the previous x
			x1, y1 = last_x, args[0]
		elif cmd == "Z":
			# Close path, return to the starting point
			x1, y1 = last_x, last_y
		else:
			continue  # If command is not recognized, skip (e.g., `m`, `l`, `z`, etc.)

		# Update the area for winding order
		# Area formula: (x1 * y2 - x2 * y1)
		area += last_x * y1 - x1 * last_y

		# Update the last coordinates to the current ones
		last_x, last_y = x1, y1
	
	return area

def check_visibility(elem, parent_transparency):
	# Check fill and opacity for transparency/visibility
	fill = elem.get("fill", "undefined")
	if fill == "undefined":
		fill = parent_transparency["fill"]
	base_opacity = float(elem.get("opacity", -1))
	if base_opacity == -1:
		opacity = parent_transparency["opacity"]
	else:
		opacity = base_opacity * parent_transparency["opacity"]
	visibility = "invisible"  # Default visibility

	# Determine transparency status
	if fill != "none" and opacity == 1.0:
		visibility = "visible"
	elif fill != "none" and opacity < 1.0:
		visibility = "border"
	elif fill == "none" or opacity == 0.0:
		visibility = "invisible"

	return visibility

def apply_masks_to_element(element):
	masked_element = element
	return [masked_element]

def convert_commands(commands):
	current_x, current_y = 0, 0  # Starting position (you can adjust if there's an initial position)
	prev_command = None  # To store the last command processed
	prev_control_x, prev_control_y = None, None
	converted_commands = []

	for command in commands:
		cmd = command["cmd"]
		args = command["args"]
		
		if cmd.upper() == "M":  # Relative move, adjust coordinates based on the current position
			# Parse arguments
			end_x, end_y = args
			if cmd == "m":
				end_x += current_x
				end_y += current_y
			# Update information for the next command
			current_x, current_y, prev_command = end_x, end_y, command["cmd"]
			# Set converted information
			converted_commands.append({"cmd": "M", "args": [end_x, end_y]})
			
		elif cmd.upper() in ["L", "H", "V"]:  # Relative line, convert to absolute
			# Parse arguments
			if cmd.upper() == "H":
				end_x = args[0]
				end_y = current_y
			elif cmd.upper() == "V":
				end_x = current_x
				end_y = args[0]
			else:
				end_x, end_y = args
			if cmd in ["l", "h", "v"]:
				end_x += current_x
				end_y += current_y
			# Update information for the next command
			current_x, current_y, prev_command = end_x, end_y, command["cmd"]
			# Set converted information
			converted_commands.append({"cmd": "L", "args": [end_x, end_y]})
			
		elif cmd.upper() == "C":  # Relative cubic Bezier curve, convert to absolute
			# Parse arguments
			control1_x, control1_y, control2_x, control2_y, end_x, end_y = args
			if cmd == "c":
				control1_x += current_x
				control1_y += current_y
				control2_x += current_x
				control2_y += current_y
				end_x += current_x
				end_y += current_y
			# Update information for the next command
			current_x, current_y, prev_control_x, prev_control_y, prev_command = end_x, end_y, control2_x, control2_y, command["cmd"]
			# Set converted information
			converted_commands.append({"cmd": "C", "args": [control1_x, control1_y, control2_x, control2_y, end_x, end_y]})
			
		elif cmd in ["S", "s"]:  # Cubic Bezier curves
			# Parse arguments
			control2_x, control2_y, end_x, end_y = args
			if cmd == "s": # Relative
				control2_x += current_x
				control2_y += current_y
				end_x += current_x
				end_y += current_y
			if prev_command.upper() in ["C", "S"]:
				control1_x = 2 * current_x - prev_control_x
				control1_y = 2 * current_y - prev_control_y
			else:
				control1_x, control1_y = current_x, current_y
			# Update information for the next command
			current_x, current_y, prev_control_x, prev_control_y, prev_command = end_x, end_y, control2_x, control2_y, command["cmd"]
			# Set converted information
			converted_commands.append({"cmd": "C", "args": [control1_x, control1_y, control2_x, control2_y, end_x, end_y]})
			
		elif cmd.upper() in ["Q", "T"]: # Quadratic Bezier curves
			# Parse arguments
			if cmd.upper == "Q":
				control_x, control_y, end_x, end_y = args
				if cmd == "q":
					control_x += current_x
					control_y += current_y
					end_x += current_x
					end_y += current_y
			else: # Smooth
				end_x, end_y = args
				if cmd == "t": #relative
					end_x += current_x
					end_y += current_y
				if prev_command.upper() in ["Q", "T"]:
					control_x = 2 * current_x - prev_control_x
					control_y = 2 * current_y - prev_control_y
				else:
					control_x, control_y = current_x, current_y
			control1_x = current_x + (2 / 3) * (control_x - current_x)
			control1_y = current_y + (2 / 3) * (control_y - current_y)
			control2_x = end_x + (2 / 3) * (control_x - end_x)
			control2_y = end_y + (2 / 3) * (control_y - end_y)
			# Update information for the next line
			current_x, current_y, prev_control_x, prev_control_y, prev_command = end_x, end_y, control2_x, control2_y, command["cmd"]
			# Set converted information
			converted_commands.append({"cmd": "C", "args": [control1_x, control1_y, control2_x, control2_y, end_x, end_y]})

		elif cmd.upper() == "A":  # Arcs
			# Parse arguments
			rx, ry, x_axis_rotation, large_arc_flag, sweep_flag, end_x, end_y = args
			if cmd == "a":  # Relative
				end_x += current_x
				end_y += current_y
			arc_commands = arc_to_cubic(rx, ry, x_axis_rotation, large_arc_flag, sweep_flag, end_x, end_y, current_x, current_y)
			# Update information for the next line
			current_x, current_y, prev_command = end_x, end_y, control2_x, control2_y, command["cmd"]
			# Set converted information
			converted_commands.extend(arc_commands)

		elif cmd.upper() == "Z":
			converted_commands.append({"cmd": "Z", "args": []})
			
	return converted_commands

def process_element(elem, parent_transformations, parent_transparency, parent_masks):
	"""Function to process an individual element and return its element(s)."""
	global path_count, rect_count, circle_count, ellipse_count, polygon_count
	tag = elem.tag.replace("{http://www.w3.org/2000/svg}", "")  # Remove namespace

	transform_attr = elem.get("transform")
	transformations = (
		parent_transformations + parse_transformations(transform_attr)
		if transform_attr
		else parent_transformations
	)
	elements = []

	# Placeholder for masking code:
	masks = []

	if tag == "path":
		d_attr = elem.get("d")
		if not d_attr:
			print(f"Skipping a <path> without 'd' attribute (path_{path_count}).")
			return []
		
		# Assign a unique identifier to the parent path
		parent_id = f"path_{path_count}"
		path_count += 1

		# Parse commands and split into sub-elements.
		commands = parse_path_commands(d_attr)
		current_element = []
		subpath_count = 1  # Tracks unique IDs for subpaths within a path

		# Determine visibility based on fill and opacity
		visibility = check_visibility(elem, parent_transparency)
		
		for command in commands:
			if command["cmd"] in ["M", "m"]:
				# If current_element is not empty, a new moveto indicates a new element.
				if current_element:
					# Calculate winding order for the current subpath
					winding_order = calculate_winding_order(current_element)
					overlap_type = "merge" if winding_order > 0 else "difference"
					
					elements.append({
						"element_id": f"{parent_id}_{subpath_count}",
						"commands": current_element,
						"transformations": transformations,
						"overlap_type": overlap_type,
						"visibility": visibility,
						"masks": masks
					})
					subpath_count += 1
					current_element = []
			current_element.append(command)

		if current_element:
			# Calculate winding order for the last subpath
			winding_order = calculate_winding_order(current_element)
			overlap_type = "merge" if winding_order > 0 else "difference"
			
			elements.append({
				"element_id": f"{parent_id}_{subpath_count}",
				"commands": current_element,
				"transformations": transformations,
				"overlap_type": overlap_type,
				"visibility": visibility,
				"masks": masks
			})
		
	elif tag == "rect":
		# Extract rectangle properties
		x = float(elem.get("x", 0))
		y = float(elem.get("y", 0))
		width = float(elem.get("width", 0))
		height = float(elem.get("height", 0))
		rx = elem.get("rx")
		ry = elem.get("ry")

		# Determine visibility based on fill and opacity
		visibility = check_visibility(elem, parent_transparency)

		if rx is None and ry is None:
			rx, ry = 0, 0
		elif rx is None:
			rx = float(ry)
		elif ry is None:
			ry = float(rx)
		else:
			rx, ry = float(rx), float(ry)

		# Ensure rx/ry don't exceed half the width/height
		rx = min(rx, width / 2)
		ry = min(ry, height / 2)

		parent_id = f"rect_{rect_count}"
		rect_count += 1

		if rx == 0 and ry == 0:
			# Simple rectangle (no rounding)
			commands = [
				{"cmd": "M", "args": [x, y]},
				{"cmd": "L", "args": [x + width, y]},
				{"cmd": "L", "args": [x + width, y + height]},
				{"cmd": "L", "args": [x, y + height]},
				{"cmd": "Z", "args": []}
			]
		else:
			# Rounded rectangle using quadratic Bézier curves
			commands = [
				{"cmd": "M", "args": [x + rx, y]},  # Move to start of top-left curve
				{"cmd": "L", "args": [x + width - rx, y]},  # Top straight line
				{"cmd": "Q", "args": [x + width, y, x + width, y + ry]},  # Top-right curve
				{"cmd": "L", "args": [x + width, y + height - ry]},  # Right straight line
				{"cmd": "Q", "args": [x + width, y + height, x + width - rx, y + height]},  # Bottom-right curve
				{"cmd": "L", "args": [x + rx, y + height]},  # Bottom straight line
				{"cmd": "Q", "args": [x, y + height, x, y + height - ry]},  # Bottom-left curve
				{"cmd": "L", "args": [x, y + ry]},  # Left straight line
				{"cmd": "Q", "args": [x, y, x + rx, y]},  # Top-left curve
				{"cmd": "Z", "args": []}  # Close path
			]

		elements.append({
			"element_id": f"{parent_id}_{rect_count}",
			"commands": commands,
			"transformations": transformations,
			"overlap_type": "merge",
			"visibility": visibility,
			"masks": masks
		})

	elif tag == "circle":
		# Extract circle properties
		cx, cy = float(elem.get("cx", 0)), float(elem.get("cy", 0))
		r = float(elem.get("r", 0))

		# Determine visibility based on fill and opacity
		visibility = check_visibility(elem, parent_transparency)

		parent_id = f"circle_{circle_count}"
		circle_count += 1

		# Control point offset factor
		k = 0.5522847498 * r

		commands = [
			{"cmd": "M", "args": [cx, cy - r]},  # Move to top
			{"cmd": "C", "args": [cx + k, cy - r, cx + r, cy - k, cx + r, cy]},  # Top-right
			{"cmd": "C", "args": [cx + r, cy + k, cx + k, cy + r, cx, cy + r]},  # Bottom-right
			{"cmd": "C", "args": [cx - k, cy + r, cx - r, cy + k, cx - r, cy]},  # Bottom-left
			{"cmd": "C", "args": [cx - r, cy - k, cx - k, cy - r, cx, cy - r]},  # Top-left
			{"cmd": "Z", "args": []}  # Close path
		]

		elements.append({
			"element_id": f"{parent_id}_{circle_count}",
			"commands": commands,
			"transformations": transformations,
			"overlap_type": "merge",
			"visibility": visibility,
			"masks": masks
		})

	elif tag == "ellipse":
		# Extract ellipse properties
		cx, cy = float(elem.get("cx", 0)), float(elem.get("cy", 0))
		rx, ry = float(elem.get("rx", 0)), float(elem.get("ry", 0))

		# Determine visibility based on fill and opacity
		visibility = check_visibility(elem, parent_transparency)

		parent_id = f"ellipse_{ellipse_count}"
		ellipse_count += 1

		kx, ky = 0.5522847498 * rx, 0.5522847498 * ry

		commands = [
			{"cmd": "M", "args": [cx, cy - ry]},
			{"cmd": "C", "args": [cx + kx, cy - ry, cx + rx, cy - ky, cx + rx, cy]},
			{"cmd": "C", "args": [cx + rx, cy + ky, cx + kx, cy + ry, cx, cy + ry]},
			{"cmd": "C", "args": [cx - kx, cy + ry, cx - rx, cy + ky, cx - rx, cy]},
			{"cmd": "C", "args": [cx - rx, cy - ky, cx - kx, cy - ry, cx, cy - ry]},
			{"cmd": "Z", "args": []}
		]

		elements.append({
			"element_id": f"{parent_id}_{ellipse_count}",
			"commands": commands,
			"transformations": transformations,
			"overlap_type": "merge",
			"visibility": visibility,
			"masks": masks
		})

	elif tag == "polygon":
		# Extract polygon properties
		points_str = elem.get("points")
		if not points_str:
			print(f"Skipping <polygon> without 'points' attribute (polygon_{polygon_count}).")
			return []

		# Determine visibility based on fill and opacity
		visibility = check_visibility(elem, parent_transparency)

		# Parse points
		points = []
		for pt in points_str.strip().split():
			coords = pt.split(',')
			if len(coords) >= 2:
				points.append((float(coords[0]), float(coords[1])))
		if len(points) < 3:
			print(f"Skipping <polygon> with insufficient points (polygon_{polygon_count}).")
			return []

		parent_id = f"polygon_{polygon_count}"
		polygon_count += 1

		commands = []
		# Move to the first point.
		commands.append({"cmd": "M", "args": [points[0][0], points[0][1]]})
		# Line to each subsequent point.
		for pt in points[1:]:
			commands.append({"cmd": "L", "args": [pt[0], pt[1]]})
		# Close the polygon.
		commands.append({"cmd": "Z", "args": []})

		elements.append({
			"element_id": f"{parent_id}_{polygon_count}",
			"commands": commands,
			"transformations": transformations,
			"overlap_type": "merge",
			"visibility": visibility,
			"masks": masks
		})

	return elements

def process_groups(elem, parent_transformations, parent_transparency, parent_masks):
	"""Process elements inside groups recursively."""
	tag = elem.tag.replace("{http://www.w3.org/2000/svg}", "")  # Remove namespace
	elements = []  # To hold the processed elements for this group
	# Do not render tags in definions
	if tag == "defs":
		return []
	elif tag in {"g", "svg", "clipPath", "mask"}:
		# Process group transformations
		transform_attr = elem.get("transform")
		new_parent_transformations = list(parent_transformations)
		if transform_attr:
			new_parent_transformations.extend(parse_transformations(transform_attr))

		# Process group transparency
		fill = elem.get("fill", "undefined")
		if fill == "undefined":
			fill = parent_transparency["fill"]
		base_opacity = float(elem.get("opacity", -1))
		opacity = 1.0 #Default opacity
		if base_opacity == -1:
			opacity = parent_transparency["opacity"]
		else:
			opacity = base_opacity * parent_transparency["opacity"]
		new_parent_transparency = {
			"fill": fill,
			"opacity": opacity
		}

		# Placeholder for masking logic
		new_parent_masks = parent_masks

		# Process all child elements of the group
		for child in elem:
			elements.extend(process_groups(child, new_parent_transformations, new_parent_transparency, new_parent_masks))  # Recursively process child elements
	else:
		# Process non-group elements (paths, rectangles, etc.) within this group
		elements.extend(process_element(elem, parent_transformations, parent_transparency, parent_masks))  # Use the group transformations here

	return elements

#Refactored parts of convert_to_mse():

def extract_elements(svg_file):
	"""
	Extracts path and rectangle elements from an SVG file, preserving SVG order.

	Each path is broken into elements, where an element is a sequence of commands 
	starting with a moveto (M/m) and continuing until the next moveto.

	Each rectangle is converted into a path-like format with four lines.

	Returns a list of elements, where each element is a dictionary containing:
	- `parent_id`: The unique identifier of the parent shape.
	- `element_id`: The unique identifier of this element within the shape.
	- `path_type`: The classification of the shape (e.g., "path", "rectangle").
	- `commands`: A list of command dictionaries for the element.
	"""
	try:
		tree = ET.parse(svg_file)
		root = tree.getroot()

		all_elements = []
		default_transparency = {
			"fill": "none",
			"opacity": 1.0
		}

		# Start processing the SVG tree from the root
		for elem in root:
			all_elements.extend(process_groups(elem, [], default_transparency, []))  # Process the root element (groups and non-groups)
		return all_elements

	except ET.ParseError as e:
		print(f"Error parsing SVG file: {e}")
		return None
	except Exception as e:
		print(f"Unexpected error: {e}")
		return None

def apply_transformations(x, y, transformations):
	"""
	Applies transformations to a single (x, y) point.
	"""
	point = np.array([x, y, 1])  # Homogeneous coordinates

	for transform in reversed(transformations):  # Reverse order for correct application
		t_type, args = transform["type"], transform["args"]

		if t_type == "translate":
			tx, ty = args[0], args[1] if len(args) > 1 else 0
			matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

		elif t_type == "scale":
			sx, sy = args[0], args[1] if len(args) > 1 else args[0]
			matrix = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

		elif t_type == "rotate":
			angle = np.radians(args[0])
			if len(args) == 3:  # Rotate around (cx, cy)
				cx, cy = args[1], args[2]
				translation1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
				rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
									 [np.sin(angle), np.cos(angle), 0],
									 [0, 0, 1]])
				translation2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
				matrix = translation2 @ rotation @ translation1
			else:
				matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
								   [np.sin(angle), np.cos(angle), 0],
								   [0, 0, 1]])

		elif t_type == "skewX":
			angle = np.radians(args[0])
			matrix = np.array([[1, np.tan(angle), 0], [0, 1, 0], [0, 0, 1]])

		elif t_type == "skewY":
			angle = np.radians(args[0])
			matrix = np.array([[1, 0, 0], [np.tan(angle), 1, 0], [0, 0, 1]])

		elif t_type == "matrix":
			if len(args) == 6:
				a, b, c, d, e, f = args
				matrix = np.array([[a, c, e], [b, d, f], [0, 0, 1]])
			else:
				continue  # Invalid matrix

		else:
			continue  # Unsupported transformation

		point = matrix @ point  # Apply transformation

	return point[0], point[1]

def apply_transformations_to_commands(commands, transformations):
	"""
	Applies transformations to each coordinate in a command.
	"""
	transformed_commands = []
	for command in commands:
		cmd = command["cmd"]
		args = command["args"]
		if cmd == "M":
			end_x, end_y = args
			new_end_x, new_end_y = apply_transformations(end_x, end_y, transformations)
			transformed_commands.append({"cmd": "M", "args": [new_end_x, new_end_y]})
		elif cmd == "L":
			end_x, end_y = args
			new_end_x, new_end_y = apply_transformations(end_x, end_y, transformations)
			transformed_commands.append({"cmd": "L", "args": [new_end_x, new_end_y]})
		elif cmd == "C":
			control1_x, control1_y, control2_x, control2_y, end_x, end_y = args
			new_end_x, new_end_y = apply_transformations(end_x, end_y, transformations)
			new_control1_x, new_control1_y = apply_transformations(control1_x, control1_y, transformations)
			new_control2_x, new_control2_y = apply_transformations(control2_x, control2_y, transformations)
			transformed_commands.append({"cmd": "C", "args": [new_control1_x, new_control1_y, new_control2_x, new_control2_y, new_end_x, new_end_y]})
		elif cmd == "Z":
			transformed_commands.append({"cmd": "Z", "args": []})
	return transformed_commands

def build_segments(element):
	"""
	Build a list of segments from a element.
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
	start_point = None  # Store the first point of a element
	commands = element["commands"]

	for command in commands:
		cmd = command["cmd"]
		args = command["args"]

		if cmd == "M":
			current_point = (args[0], args[1])
			start_point = current_point  # Store element start

		elif cmd == "L":
			if current_point is None:
				continue
			# For L, we expect pairs of coordinates.
			# Process them in pairs.
			for i in range(0, len(args), 2):
				end = (args[i], args[i+1])
				segments.append(("L", current_point, end))
				current_point = end

		elif cmd == "C":
			if current_point is None:
				continue
			c1 = (args[0], args[1])
			c2 = (args[2], args[3])
			end = (args[4], args[5])
			segments.append(("C", current_point, c1, c2, end))
			current_point = end

		elif cmd == "Z":
			if current_point is not None and start_point is not None and current_point != start_point:
				segments.append(("L", current_point, start_point))
				current_point = start_point  # Ensure closure

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
	Converts an SVG file's paths into MSE format, handling multiple elements.
	Supports M/m, C/c, and L/l commands; others are skipped.
	"""
	viewbox = get_viewbox(svg_file)
	if not viewbox:
		return "Error: No valid viewBox found."

	elements = extract_elements(svg_file)
	if not elements:
		return "Error: No valid elements found."

	mse_output = "mse_version: 0.3.5"

	for element in reversed(elements):
		# Get the visibility of the element, defaulting to "invisible" if not set
		visibility = element.get("visibility", "invisible")

		# Skip if the visibility is "invisible"
		if visibility == "invisible":
			continue  # Skip processing this element

		# Get the overlap type (this is already part of the element)
		overlap_type = element.get("overlap_type", "merge")  # Default to "merge" if not set
		
		if visibility == "visible" and overlap_type == "merge":
			combine_value = "merge"
		elif visibility == "border" and overlap_type == "merge":
			combine_value = "border"
		elif overlap_type == "difference":
			combine_value = "difference"
		else:
			# Default to overlap if no other conditions match
			combine_value = "overlap"

		# Convert commands before building segments
		if "commands" in element:
			element["commands"] = convert_commands(element["commands"])
			if "transformations" in element:
				element["commands"] = apply_transformations_to_commands(element["commands"], element["transformations"])

		masked_elements = apply_masks_to_element(element)
		
		for masked_element in masked_elements:
			segments = build_segments(masked_element)
			if not segments:
				continue

		element_name = element["element_id"] if "element_id" in element else "unnamed_path"

		mse_output += f"""
part:
	type: shape
	name: {element_name}
	combine: {combine_value}
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