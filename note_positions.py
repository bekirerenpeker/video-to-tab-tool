from template_remover import remove_all_templates
import numpy as np
import cv2
import os
import random

DEBUG=True

def remove_vertical_bars(thresh_img, avg_spacing, height_multiplier=2.2):
    pre_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    dilated_for_detection = cv2.dilate(thresh_img, pre_kernel, iterations=1)

    bar_kernel_height = int(avg_spacing * height_multiplier)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, bar_kernel_height))
    
    detected_bars = cv2.morphologyEx(dilated_for_detection, cv2.MORPH_OPEN, vertical_kernel)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    detected_bars = cv2.dilate(detected_bars, dilation_kernel, iterations=1)
    
    clean_img = cv2.subtract(thresh_img, detected_bars)
    return clean_img, detected_bars

def detect_and_remove_hammer_ons_pull_offs(frame, string_y_positions):
    def get_arch_data(contour, min_width=15):
        pts = contour.reshape(-1, 2)
        x_pts = pts[:, 0].astype(float)
        y_pts = pts[:, 1].astype(float)

        if len(x_pts) < 10: return None
        width = np.max(x_pts) - np.min(x_pts)
        if width < min_width: return None

        try:
            coeffs, residuals, rank, _, _ = np.polyfit(x_pts, y_pts, 2, full=True)
            a, b, c = coeffs

            if len(residuals) > 0:
                mse = residuals[0] / len(x_pts)
                if mse > 7.0: return None
            
            orientation = "up" if a < 0 else "down"
            
            return {
                "bbox": cv2.boundingRect(contour),
                "contour": contour,
                "orientation": orientation
            }

        except Exception as e:
            print(f"Fit error: {e}")
            return None 

    # enlarge the shapes horzontally so the shapes always connect
    heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 2))
    healed = cv2.dilate(frame.copy(), heal_kernel, iterations=1)
    contours, _ = cv2.findContours(healed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    arches_data = [e for e in [get_arch_data(c) for c in contours] if e != None]
    cv2.drawContours(frame, [cd["contour"] for cd in arches_data], -1, (0, 0, 0), 6)

    string_y_positions = np.array(string_y_positions)
    hopo_data = [[] for _ in range(6)]
    for data in arches_data:
            x, y, w, h = data["bbox"]
            center_x = x + (w // 2)
            center_y = y + (h // 2)
            orientation = data["orientation"]

            if orientation == "up":
                valid_indices = np.where(string_y_positions < center_y)[0]
                if len(valid_indices) > 0:
                    idx = valid_indices[np.argmax(string_y_positions[valid_indices])]
                    hopo_data[idx].append((center_x, data["bbox"], data["orientation"]))
            else: 
                valid_indices = np.where(string_y_positions > center_y)[0]
                if len(valid_indices) > 0:
                    idx = valid_indices[np.argmin(string_y_positions[valid_indices])]
                    hopo_data[idx].append((center_x, data["bbox"], data["orientation"]))

    return hopo_data
   
def preprocess_for_numbers(frame, avg_spacing):
    if DEBUG and not (os.listdir("debug_density") if os.path.exists("debug_density") else os.makedirs("debug_density")):
        pass

    processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_h, img_w = processed.shape
    if DEBUG: cv2.imwrite("debug_density/0_grayscale.png", processed)

    # STANDARDIZE TO WHITE-ON-BLACK
    median_brightness = np.median(processed)
    is_light_bg = median_brightness > 127
    
    line_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    if is_light_bg: processed = cv2.morphologyEx(processed, cv2.MORPH_BLACKHAT, line_kernel_h)
    else: processed = cv2.morphologyEx(processed, cv2.MORPH_TOPHAT, line_kernel_h)
    if DEBUG: cv2.imwrite("debug_density/01_numbers_only.png", processed)

    # REMOVE TEMPLATES
    processed = remove_all_templates(processed, avg_spacing)
    if DEBUG: cv2.imwrite("debug_density/01_templates_removed.png", processed)

    # REMOVE GRAY VALUES (comment out if the tab doesnt have good contrast)
    _, strict_mask = cv2.threshold(processed, 150, 255, cv2.THRESH_BINARY)
    processed = cv2.bitwise_and(processed, processed, mask=strict_mask)
    if DEBUG: cv2.imwrite("debug_density/02_cleaned_gray_noise.png", processed)

    # REMOVE VERTICAL BARS
    processed, detected_bars = remove_vertical_bars(processed, avg_spacing)
    if DEBUG:
        cv2.imwrite("debug_density/03_bars_removed.png", processed)
        # cv2.imwrite("debug_density/03_only_bars.png", detected_bars)

    # BOLD THE NUMBERS SLIGHTLY TO ENSURE '0' AND '8' STAY CONNECTED
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.dilate(processed, dilation_kernel, iterations=2)
    _, processed = cv2.threshold(processed, 40, 255, cv2.THRESH_BINARY)
    if DEBUG: cv2.imwrite("debug_density/04_bolded.png", processed)

    return processed

def detect_shape_bboxes(frame):
    contours, _ = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    img_h, img_w = frame.shape[:2] 
    padding = 1
    bboxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        new_x = max(0, x - padding)
        new_y = max(0, y - padding)
        new_w = min(img_w - new_x, w + 2 * padding)
        new_h = min(img_h - new_y, h + 2 * padding)
        bboxes.append((new_x, new_y, new_w, new_h))
        
    return bboxes

def map_shapes_to_strings(bboxes, string_y_positions):
    avg_spacing = abs(string_y_positions[0] - string_y_positions[-1]) / 5
    notes = [[] for _ in range(6)]

    # split merged notes
    split = []
    for i, (x, y, w, h) in enumerate(bboxes):
        if h < avg_spacing * 1.3: continue
        note_count = int(round((h / avg_spacing)))
        split_boxes = [(x, y+(i*h//note_count), w, h//note_count) for i in range(note_count)]
        bboxes[i] = split_boxes[0]
        for b in split_boxes[1:]: bboxes.append(b)
        split.extend(split_boxes)
        
    for x, y, w, h in bboxes:
        center_x, center_y = x + (w // 2), y + (h // 2)
        distances = np.abs(center_y - np.array(string_y_positions))
        min_dist = np.min(distances)
        if min_dist > avg_spacing * 0.3: continue 
        closest_string_index = distances.argmin()
        notes[closest_string_index].append((center_x, (x, y, w, h)))
    
    return notes

def merge_close_points(notes, min_dist=10):
    def calculate_union_box(group):
        x_coords = [item[1][0] for item in group]
        y_coords = [item[1][1] for item in group]
        r_coords = [item[1][0] + item[1][2] for item in group]
        b_coords = [item[1][1] + item[1][3] for item in group]
        
        u_x = min(x_coords)
        u_y = min(y_coords)
        u_w = max(r_coords) - u_x
        u_h = max(b_coords) - u_y

        u_center_x = u_x + (u_w // 2)
        return (u_center_x, [u_x, u_y, u_w, u_h])

    cleaned_all_notes = []

    for string_data in notes:
        if not string_data:
            cleaned_all_notes.append([])
            continue
        
        merged = []
        string_data.sort(key=lambda item: item[1][0])
        current_group = [string_data[0]]
        
        for i in range(1, len(string_data)):
            group_right_edge = max(item[1][0] + item[1][2] for item in current_group)
            curr_left_edge = string_data[i][1][0]
            edge_gap = curr_left_edge - group_right_edge
            
            if edge_gap < min_dist:
                current_group.append(string_data[i])
            else:
                merged.append(calculate_union_box(current_group))
                current_group = [string_data[i]]
        
        if current_group: 
            merged.append(calculate_union_box(current_group))
            
        merged.sort(key=lambda item: item[0])
        cleaned_all_notes.append(merged)
        
    return cleaned_all_notes

def detect_notes(frame, string_y_positions):
    avg_spacing = abs(string_y_positions[0] - string_y_positions[-1]) / 5
    processed = preprocess_for_numbers(frame, avg_spacing)
    arches = detect_and_remove_hammer_ons_pull_offs(processed, string_y_positions)
    bboxes = detect_shape_bboxes(processed)
    notes = map_shapes_to_strings(bboxes, string_y_positions)
    merged_notes = merge_close_points(notes)
    return merged_notes, arches