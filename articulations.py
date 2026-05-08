import numpy as np
import cv2
import random

DEBUG=False

# TODO: add down and up stroke detection and removal

def is_arpeggio(contour, avg_spacing, frame):
    x, y, w, h = cv2.boundingRect(contour)
    if h < avg_spacing * 2.0: return False
    if w < avg_spacing * 0.12 or w > avg_spacing * 1.5: return False

    roi = frame[y:y+h, x:x+w]
    
    # Find the average X-pixel for every row
    centers = []
    for row in roi:
        on_pixels = np.where(row > 0)[0]
        if len(on_pixels) > 0: centers.append(np.mean(on_pixels))
    
    # Smooth the signal slightly to ignore tiny pixel jitters
    window = 3
    smoothed_centers = np.convolve(centers, np.ones(window)/window, mode='valid')
    
    local_avg = np.mean(smoothed_centers)
    crossings = 0
    for i in range(1, len(smoothed_centers)):
        if (smoothed_centers[i-1] < local_avg and smoothed_centers[i] >= local_avg) or \
        (smoothed_centers[i-1] > local_avg and smoothed_centers[i] <= local_avg):
            crossings += 1

    return crossings >= 4 and (crossings / len(centers)) >= 0.13

def detect_and_remove_arp_strokes(frame, string_y_positions):
    heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    healed = cv2.dilate(frame.copy(), heal_kernel, iterations=1)
    contours, _ = cv2.findContours(healed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    avg_spacing = abs(string_y_positions[0] - string_y_positions[-1]) / 5
    arps = [e for e in contours if is_arpeggio(e, avg_spacing, frame)]

    arps_data = []
    for a in arps:
        x, y, w, h = cv2.boundingRect(a)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)

        padding = int(avg_spacing * 0.1)
        arp_top, arp_bottom = y - padding, y + h + padding
        contained_strings = [
            i for i, s_y in enumerate(string_y_positions) 
            if s_y >= arp_top and s_y <= arp_bottom
        ]
        if contained_strings:
            top_string_idx = min(contained_strings)
            bottom_string_idx = max(contained_strings)
        else: # Fallback: If it's too small to contain a string, use the single closest
            top_string_idx = min(range(len(string_y_positions)), key=lambda i: abs(string_y_positions[i] - arp_top))
            bottom_string_idx = top_string_idx

        arps_data.append((x + (w//2), top_string_idx, bottom_string_idx))

    if DEBUG:
        debug_frame = healed.copy()
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)
        for c in arps:
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            cv2.drawContours(debug_frame, [c], -1, color, 1)
        cv2.imshow("Arpeggio Strokes", debug_frame)
    
    return arps_data

def detect_and_remove_vertical_bars(frame, string_y_positions):
    import random
    
    avg_spacing = abs(string_y_positions[0] - string_y_positions[-1]) / 5
    heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    healed = cv2.dilate(frame.copy(), heal_kernel, iterations=1)

    # ISOLATION: Destroy crossing lines, arches, and digits
    vert_kernel_height = int(avg_spacing * 4.5)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_height))
    vertical_spines = cv2.morphologyEx(healed, cv2.MORPH_OPEN, vert_kernel)

    # Find contours on the isolated spines (no crossing lines attached anymore!)
    contours, _ = cv2.findContours(vertical_spines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    only_bars = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < avg_spacing * 0.12: only_bars.append((x, y, w, h))

    p = int(avg_spacing * 0.2)
    for x, y, w, h in only_bars:
        cv2.rectangle(frame, (x - p, y - p), (x + w + p, y + h + p), (0, 0, 0), -1)

    if DEBUG:
        debug_frame = cv2.cvtColor(healed.copy(), cv2.COLOR_GRAY2BGR)
        for x, y, w, h in only_bars:
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            cv2.rectangle(debug_frame, (x - p, y - p), (x + w + p, y + h + p), color, 1)
        cv2.imshow("Isolated Vertical Spines", debug_frame)

    return [(x + (w // 2)) for x, y, w, h in only_bars]

# TODO: improve detection by checking the head (min or max point) is in the middle
# this will reduce the false detections and wont detect the arches that are not fully showing and cut by the frame
def get_arch_data(contour, avg_spacing):
    pts = contour.reshape(-1, 2)
    x_pts = pts[:, 0].astype(float)
    y_pts = pts[:, 1].astype(float)

    if len(x_pts) < 10: return None
    width = np.max(x_pts) - np.min(x_pts)
    if width < avg_spacing * 1.2: return None

    try:
        coeffs, residuals, rank, _, _ = np.polyfit(x_pts, y_pts, 2, full=True)
        a, b, c = coeffs

        if len(residuals) > 0:
            mse = residuals[0] / len(x_pts)
            if mse > 10.0: return None
        
        orientation = "up" if a < 0 else "down"
        
        return {
            "bbox": cv2.boundingRect(contour),
            "contour": contour,
            "orientation": orientation
        }

    except Exception as e:
        print(f"Fit error: {e}")
        return None 

def detect_and_remove_hammer_ons_pull_offs(frame, string_y_positions):
    # enlarge the shapes horzontally so the shapes always connect
    heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    healed = cv2.dilate(frame.copy(), heal_kernel, iterations=1)
    contours, _ = cv2.findContours(healed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    avg_spacing = abs(string_y_positions[0] - string_y_positions[-1]) / 5
    arches_data = [e for e in [get_arch_data(c, avg_spacing) for c in contours] if e != None]
    cv2.drawContours(frame, [cd["contour"] for cd in arches_data], -1, (0, 0, 0), 6)

    if DEBUG:
        debug_frame = healed.copy()
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)
        for i, cnt in enumerate([cd["contour"] for cd in arches_data]):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.drawContours(debug_frame, cnt, -1, color, 2)
        cv2.imshow("Hammer On / Pull Off", debug_frame)

    string_y_positions = np.array(string_y_positions)
    hopo_data = [[] for _ in range(6)]
    for data in arches_data:
        x, y, w, h = data["bbox"]
        center_x = x + (w // 2)
        center_y = y + (h // 2)
        orientation = data["orientation"]
        str_idx = -1

        dist = float('inf')
        if orientation == "up":
            valid_indices = np.where(string_y_positions < y)[0]
            if len(valid_indices) > 0:
                str_idx = valid_indices[np.argmax(string_y_positions[valid_indices])]
                dist = abs(string_y_positions[str_idx] - (y))
        else: 
            valid_indices = np.where(string_y_positions > y+h)[0]
            if len(valid_indices) > 0:
                str_idx = valid_indices[np.argmin(string_y_positions[valid_indices])]
                dist = abs(string_y_positions[str_idx] - (y+h))

        if dist > (avg_spacing*1.2): continue
        if str_idx == -1 or abs(string_y_positions[str_idx] - center_y) > avg_spacing * 1.5: continue
        hopo_data[str_idx].append((center_x, data["bbox"], data["orientation"]))

    return hopo_data

def get_line_data(contour, min_width=7, min_height=7):
    pts = contour.reshape(-1, 2)
    x_pts = pts[:, 0].astype(float)
    y_pts = pts[:, 1].astype(float)

    if len(x_pts) < 5: return None
    width = np.max(x_pts) - np.min(x_pts)
    height = np.max(y_pts) - np.min(y_pts)
    if width < min_width or height < min_height: return None
    
    try:
        coeffs, residuals, _, _, _ = np.polyfit(x_pts, y_pts, 1, full=True)

        slope = coeffs[0]
        #if abs(slope) < 0.3 or abs(slope) > 3.0: return None

        if len(residuals) > 0:
            mse = residuals[0] / len(x_pts)
            if mse > 5: return None
        
        return {
            "contour": contour,
            "bbox": cv2.boundingRect(contour),
            "orientation": "up" if slope < 0 else "down",
        }
    except:
        return None

# TODO: improve detection and achive better seperation for close slide lines and characters 
# since they can merge easily and not be detected
def detect_and_remove_slides(frame, string_y_positions):
    # enlarge in a cross pattern so slides are connected
    heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    healed = cv2.dilate(frame.copy(), heal_kernel, iterations=1)
    contours, _ = cv2.findContours(healed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    lines_data = [e for e in [get_line_data(c) for c in contours] if e != None]
    cv2.drawContours(frame, [ld["contour"] for ld in lines_data], -1, (0, 0, 0), 4)

    avg_spacing = abs(string_y_positions[0] - string_y_positions[-1]) / 5
    string_y_positions = np.array(string_y_positions)
    slides = [[] for _ in range(6)]
    for data in lines_data:
        x, y, w, h = data["bbox"]
        center_x, center_y = x + (w // 2), y + (h // 2)
        orientation = data["orientation"]
        distances = np.abs(center_y - np.array(string_y_positions))
        min_dist = np.min(distances)
        if min_dist > avg_spacing * 0.3: continue 
        closest_string_index = distances.argmin()
        slides[closest_string_index].append((center_x, (x, y, w, h), orientation))
    
    if DEBUG:
        debug_frame = healed.copy()
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)
        for i, cnt in enumerate([ld["contour"] for ld in lines_data]):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.drawContours(debug_frame, cnt, -1, color, 2)
        cv2.imshow("Slides", debug_frame)
    
    return slides
