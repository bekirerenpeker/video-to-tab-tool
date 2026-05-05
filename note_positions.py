from template_remover import remove_all_templates
import cv2
import numpy as np
import pytesseract
import os

def calibrate_strings(sample_frame):
    print("\n[ACTION] Click on each of the 6 strings (Top to Bottom).")
    y_coords = []
    
    def click_event(event, x, y, flags, param):
        display_img = sample_frame.copy()
        # Horizontal Preview Line
        cv2.line(display_img, (0, y), (sample_frame.shape[1], y), (255, 255, 0), 1)
        
        # Draw existing points
        for yc in y_coords:
            cv2.circle(display_img, (x, yc), 3, (0, 0, 255), -1)
        
        cv2.imshow("Calibrate Strings", display_img)

        if event == cv2.EVENT_LBUTTONDOWN:
            y_coords.append(y)
            print(f"String {len(y_coords)} set at Y={y}")

    cv2.imshow("Calibrate Strings", sample_frame)
    cv2.setMouseCallback("Calibrate Strings", click_event)
    
    while len(y_coords) < 6:
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cv2.destroyWindow("Calibrate Strings")
    return sorted(y_coords)

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

def preprocess_for_numbers(frame, avg_spacing, debug=True):
    if debug and not (os.listdir("debug_density") if os.path.exists("debug_density") else os.makedirs("debug_density")):
        pass

    processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_h, img_w = processed.shape
    if debug: cv2.imwrite("debug_density/0_grayscale.png", processed)

    # STANDARDIZE TO WHITE-ON-BLACK
    median_brightness = np.median(processed)
    is_light_bg = median_brightness > 127
    
    line_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    if is_light_bg: processed = cv2.morphologyEx(processed, cv2.MORPH_BLACKHAT, line_kernel_h)
    else: processed = cv2.morphologyEx(processed, cv2.MORPH_TOPHAT, line_kernel_h)
    if debug: cv2.imwrite("debug_density/01_numbers_only.png", processed)

    # REMOVE TEMPLATES
    processed = remove_all_templates(processed, avg_spacing, debug=False)
    if debug: cv2.imwrite("debug_density/01_templates_removed.png", processed)

    # REMOVE GRAY VALUES (comment out if the tab doesnt have good contrast)
    _, strict_mask = cv2.threshold(processed, 150, 255, cv2.THRESH_BINARY)
    processed = cv2.bitwise_and(processed, processed, mask=strict_mask)
    if debug: cv2.imwrite("debug_density/02_cleaned_gray_noise.png", processed)

    # REMOVE VERTICAL BARS
    processed, detected_bars = remove_vertical_bars(processed, avg_spacing)
    if debug:
        cv2.imwrite("debug_density/03_bars_removed.png", processed)
        cv2.imwrite("debug_density/03_only_bars.png", detected_bars)

    # BOLD THE NUMBERS SLIGHTLY TO ENSURE '0' AND '8' STAY CONNECTED
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.dilate(processed, dilation_kernel, iterations=1)
    _, processed = cv2.threshold(processed, 40, 255, cv2.THRESH_BINARY)
    if debug: cv2.imwrite("debug_density/04_bolded.png", processed)

    return processed

def detect_note_bboxes(sample_frame, string_y_positions, debug=True):
    avg_spacing = abs(string_y_positions[1] - string_y_positions[0])
    h_reach = int(avg_spacing * 0.4) 
    all_notes_data = [[] for _ in range(6)] # will store (x, [x, y, w, h])

    processed = preprocess_for_numbers(sample_frame, avg_spacing, debug)
    img_h, img_w = processed.shape
    note_h = int(avg_spacing * 1.2)  

    for i, y in enumerate(string_y_positions):
        y_top = max(0, y - h_reach)
        y_bot = min(img_h, y + h_reach)
        strip = processed[y_top:y_bot, :]
        
        vertical_density = np.sum(strip, axis=0) // 255
        
        # Peak Detection with a 'Gap' requirement
        min_pixel_height = 3
        in_note = False
        current_note_pixels = []
        
        for x in range(img_w):
            if vertical_density[x] >= min_pixel_height:
                if not in_note: in_note = True
                current_note_pixels.append(x)
            else:
                if len(current_note_pixels) > 2:
                    # 1. Horizontal Bounds
                    x_min = current_note_pixels[0]
                    x_max = current_note_pixels[-1]
                    width = int((x_max - x_min + 1) * 1.7)
                    center_x = x_min + (width // 2) - int(width * 0.35)

                    note_slice = strip[:, x_min:x_max+1]
                    y_indices = np.where(np.sum(note_slice, axis=1) > 0)[0]
                    
                    if len(y_indices) > 0:
                        actual_y_top = y_top + y_indices[0]
                        actual_y_bot = y_top + y_indices[-1]
                        y_center = (actual_y_top + actual_y_bot) // 2
                        bounding_box = [int(center_x - width//2), int(y_center - note_h//2), int(width), int(note_h)]
                        all_notes_data[i].append((center_x, bounding_box))

                current_note_pixels = []
                in_note = False
        
    all_notes_data[i].sort(key=lambda x: x[0])

    return all_notes_data

def merge_close_points(all_notes_data, min_dist=10):
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

    for string_data in all_notes_data:
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