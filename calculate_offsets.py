from export import save_offset_values
from export import import_tab_data
import cv2
import numpy as np
import time

# TODO: add custom edge cases of notes that the ocr mixes and give them a higher score
INF, NEG_INF = 100000, -100000
MATCH_RADIUS = 10
JUMP_INTERVAL = 10

PERFECT_MATCH_SCORE = 200 # same note
PARTIAL_MATCH_SCORE = 70 # n and note
CONFLICT_SCORE = -100 # different notes
NO_MATCH_SCORE = -200 # no match within radius
OUTSIDE_BOUNDS_SCORE = 0 # note out of bounds

def find_best_match_index(string1, string2, offset, note_idx):
    best_match_idx = -1
    best_match_score = 0 # 0 for different notes 1 for same note
    best_match_dist = INF
    
    for idx, note in enumerate(string2):
        dist = abs(note - note_idx)
        if dist > MATCH_RADIUS: continue
        
        score = 1 if string1[note_idx] == note else 0
        if score < best_match_score or (score == best_match_score and dist > best_match_dist): continue

        best_match_idx = idx
        best_match_score = score
        best_match_dist = dist
            
    return best_match_idx, best_match_score, best_match_dist

def get_offset_range_between_frames(frame1, frame2):
    all_x1 = [note[0] for string in frame1 for note in string]
    all_x2 = [note[0] for string in frame2 for note in string]
    return 0, 1280

def calculate_alignment_score(frame1, frame2, offset):
    def get_pair_weight(val1, val2):
        if val1 == val2: return PERFECT_MATCH_SCORE
        if val1 == 'N' or val2 == 'N': return PARTIAL_MATCH_SCORE
        return CONFLICT_SCORE

    total_score, active_note_count = 0, 0

    for s_idx in range(6):
        notes1 = frame1[s_idx]
        notes2 = frame2[s_idx]
        matched_in_frame2 = set()

        for x1, val1 in notes1:
            target_x = x1 - offset 
            if 0 <= target_x <= 1280: active_note_count += 1 
            
            best_match_idx = -1
            best_match_score = -999
            best_match_dist = INF

            for idx2, (x2, val2) in enumerate(notes2):
                if idx2 in matched_in_frame2: continue
                
                dist = abs(x2 - target_x)
                if dist <= MATCH_RADIUS:
                    current_pair_score = get_pair_weight(val1, val2)
                    if current_pair_score < best_match_score or (current_pair_score == best_match_score and dist > best_match_dist): 
                        continue

                    best_match_score = current_pair_score
                    best_match_dist = dist
                    best_match_idx = idx2

            if best_match_idx != -1:
                total_score += best_match_score
                matched_in_frame2.add(best_match_idx)
            else:
                if 0 <= target_x <= 1280: total_score += NO_MATCH_SCORE 
                else: total_score += OUTSIDE_BOUNDS_SCORE

        for idx2, (x2, val2) in enumerate(notes2):
            if idx2 not in matched_in_frame2:
                original_x = x2 + offset
                if 0 <= original_x <= 1280:
                    active_note_count += 1
                    total_score += NO_MATCH_SCORE 
                else: total_score += OUTSIDE_BOUNDS_SCORE
                    
    if active_note_count == 0: return 0
    return total_score / active_note_count

def calculate_best_alignment(frame1, frame2):
    offset, max_offset = get_offset_range_between_frames(frame1, frame2)
    best_score, best_offset = NEG_INF, NEG_INF
    # print(f"max offset: {max_offset}")

    while offset < max_offset:
        score = calculate_alignment_score(frame1, frame2, offset)
        # print(f"score: {score} at offset {offset}")
        if score > (best_score * 1.3):
            best_score = score
            best_offset = offset
        offset += JUMP_INTERVAL
    
    return best_score, best_offset

mouse_offset = 0
def on_mouse(event, x, y, flags, param):
    global mouse_offset
    # We use the x-coordinate of the mouse as a manual offset tester
    # Subtracting the 50px margin we use in the drawing
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_offset = max(0, x - 50)

def show_alignment_debug(frame1, frame2, best_offset, best_score, frame_idx):
    global mouse_offset
    cv2.namedWindow("Alignment Debugger")
    cv2.setMouseCallback("Alignment Debugger", on_mouse)
    
    # Initialize mouse_offset to the calculated best_offset for the first look
    mouse_offset = best_offset
    start_time = time.time()
    user_pressed_enter = False
    user_pressed_space = False

    while True:
        # Canvas: Height for 6 strings + space for info
        height, width = 600, 1400
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw string lines
        for i in range(6):
            y = 50 + (i * 50)
            cv2.line(canvas, (50, y), (width - 50, y), (200, 200, 200), 1)

        # 1. Draw Frame 1 (Red) - Static Reference
        for s_idx in range(6):
            y = 50 + (s_idx * 50)
            for x, val in frame1[s_idx]:
                cv2.circle(canvas, (int(x) + 50, y), 8, (0, 0, 255), 2)
                cv2.putText(canvas, str(val), (int(x) + 45, y - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 2. Draw "Calculated Best" (Green) - Small static markers
        for s_idx in range(6):
            y = 50 + (s_idx * 50)
            for x, val in frame2[s_idx]:
                calc_x = int(x + best_offset) + 50
                if 50 <= calc_x <= width - 50:
                    cv2.circle(canvas, (calc_x, y), 12, (255, 0, 0), 1)
                    cv2.putText(canvas, str(val), (calc_x - 5, y + 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 3. Draw "Ghost" Image (Blue) - Controlled by Mouse
        for s_idx in range(6):
            y = 50 + (s_idx * 50)
            for x, val in frame2[s_idx]:
                ghost_x = int(x + mouse_offset) + 50
                if 50 <= ghost_x <= width - 50:
                    cv2.drawMarker(canvas, (ghost_x, y), (0, 150, 0), cv2.MARKER_TILTED_CROSS, 10, 1)

        # Info Text
        cv2.putText(canvas, f"FRAME {frame_idx} -> {frame_idx+1}", (50, 400), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, f"CALCULATED BEST OFFSET: {best_offset}", (50, 430), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
        cv2.putText(canvas, f"MOUSE (GHOST) OFFSET: {int(mouse_offset)}", (50, 460), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(canvas, f"OFFSET: {best_offset}", (50, 490), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(canvas, f"SCORE: {best_score}", (50, 520), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("Alignment Debugger", canvas)
        
        key = cv2.waitKey(10) & 0xFF
        if key == 13: user_pressed_enter = True; break
        elif key == ord(' '): user_pressed_space = not user_pressed_space
        elif key == ord('q'): cv2.destroyAllWindows(); exit()
        if not user_pressed_space and (time.time() - start_time) > (0.5 if best_offset == 0 else 1.5): break

    return not user_pressed_enter

def calculate_offsets(tab_data, debug=True):
    offsets = []
    show_debug_window = debug 

    for i in range(len(tab_data) - 1):
        f1 = tab_data[i]
        f2 = tab_data[i+1]
        best_score, best_offset = calculate_best_alignment(f1, f2)
        offsets.append(best_offset)
        print(f"Pair {i}-{i+1}: Best Score {best_score} | Best Offset {best_offset}")
        if show_debug_window: show_debug_window = show_alignment_debug(f1, f2, best_offset, best_score, i)

    save_offset_values(offsets)
    return offsets

def main():
    tab_data = import_tab_data()
    offsets = calculate_offsets(tab_data)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()