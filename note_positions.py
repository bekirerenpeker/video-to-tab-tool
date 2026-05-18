from articulations import detect_and_remove_down_up_strokes
from articulations import detect_and_remove_arp_strokes
from articulations import detect_and_remove_hammer_ons_pull_offs, detect_and_remove_slides, detect_and_remove_vertical_bars
from template_remover import remove_all_templates
import numpy as np
import cv2
import os

OUTPUT_DIR = "output/debug_density"
DEBUG=True

def preprocess_for_numbers(frame, avg_spacing):
    if DEBUG and not (os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else os.makedirs(OUTPUT_DIR)):
        pass

    processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_h, img_w = processed.shape
    if DEBUG: cv2.imwrite(f"{OUTPUT_DIR}/0_grayscale.png", processed)

    # STANDARDIZE TO WHITE-ON-BLACK
    median_brightness = np.median(processed)
    is_light_bg = median_brightness > 127
    
    line_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    if is_light_bg: processed = cv2.morphologyEx(processed, cv2.MORPH_BLACKHAT, line_kernel_h)
    else: processed = cv2.morphologyEx(processed, cv2.MORPH_TOPHAT, line_kernel_h)
    if DEBUG: cv2.imwrite(f"{OUTPUT_DIR}/01_numbers_only.png", processed)

    # REMOVE TEMPLATES
    processed, detected_templates = remove_all_templates(processed, avg_spacing)
    if DEBUG: cv2.imwrite(f"{OUTPUT_DIR}/01_templates_removed.png", processed)

    # REMOVE GRAY VALUES (comment out if the tab doesnt have good contrast)
    _, strict_mask = cv2.threshold(processed, 140, 255, cv2.THRESH_BINARY)
    processed = cv2.bitwise_and(processed, processed, mask=strict_mask)
    if DEBUG: cv2.imwrite(f"{OUTPUT_DIR}/02_cleaned_gray_noise.png", processed)

    return processed, detected_templates

def detect_shape_bboxes(frame, avg_spacing):
    # BOLD THE NUMBERS SLIGHTLY TO ENSURE '0' AND '8' STAY CONNECTED
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.dilate(frame, dilation_kernel, iterations=2)
    _, processed = cv2.threshold(processed, 40, 255, cv2.THRESH_BINARY)
    if DEBUG: cv2.imwrite(f"{OUTPUT_DIR}/04_bolded.png", processed)

    contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    img_h, img_w = frame.shape[:2] 
    raw_bboxes = []
    padding = 1

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < avg_spacing * 0.3 or h < avg_spacing * 0.4: continue
        new_x = max(0, x - padding)
        new_y = max(0, y - padding)
        new_w = min(img_w - new_x, w + padding)
        new_h = min(img_h - new_y, h + padding)
        raw_bboxes.append((new_x, new_y, new_w, new_h))

    # --- REMOVE CONTAINED BOXES ---
    # Sort by area descending so we check larger boxes first
    raw_bboxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    bboxes = []

    for i, box_a in enumerate(raw_bboxes):
        is_contained = False
        ax1, ay1, aw, ah = box_a
        ax2, ay2 = ax1 + aw, ay1 + ah
        
        for j, box_b in enumerate(bboxes): # Check against boxes already kept
            bx1, by1, bw, bh = box_b
            bx2, by2 = bx1 + bw, by1 + bh

            # Calculate intersection area
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)

            if ix2 > ix1 and iy2 > iy1:
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                area_a = aw * ah
                # If more than 90% of box_a is inside box_b, it's contained
                if intersection_area / area_a > 0.9:
                    is_contained = True
                    break
        
        if not is_contained:
            bboxes.append(box_a)

    return bboxes

def map_shapes_to_strings(bboxes, string_y_positions):
    avg_spacing = abs(string_y_positions[0] - string_y_positions[-1]) / 5
    notes = [[] for _ in range(6)]
    padding = 1

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

def merge_close_points(notes, min_dist=5):
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
    processed, detected_templates = preprocess_for_numbers(frame, avg_spacing)

    arp_strokes = detect_and_remove_arp_strokes(processed, string_y_positions)
    bars = detect_and_remove_vertical_bars(processed, string_y_positions)
    strokes = detect_and_remove_down_up_strokes(processed, string_y_positions)

    arches = detect_and_remove_hammer_ons_pull_offs(processed, string_y_positions)
    slides = detect_and_remove_slides(processed, string_y_positions)

    bboxes = detect_shape_bboxes(processed, avg_spacing)
    notes = map_shapes_to_strings(bboxes, string_y_positions)
    merged_notes = merge_close_points(notes)

    return merged_notes, arches, slides, bars, strokes, arp_strokes, detected_templates