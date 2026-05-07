from export import save_final_tab
from export import read_list ,import_raw_tab_data
from sklearn.cluster import DBSCAN
import numpy as np
import cv2

DEBUG = True

def cluster_notes_to_tab(all_frames, offsets):
    global_points = []
    current_offset = 0
    
    # Track which frame each point came from for confidence scoring
    for i, frame in enumerate(all_frames):
        if i > 0:
            current_offset += offsets[i-1]
        for s_idx, string_notes in enumerate(frame):
            for x, fret in string_notes:
                # Store: [global_x, string_idx, fret_value, frame_index]
                global_points.append([x + current_offset, s_idx, fret, i])

    if not global_points: return []
    final_tab = []
    
    for s_idx in range(6):
        string_points = [p for p in global_points if p[1] == s_idx]
        if not string_points: continue
            
        data = np.array([[p[0], 0] for p in string_points])
        
        # Use a slightly larger epsilon to bridge small alignment gaps
        # Lower min_samples to 1 to stop deleting unique detections
        clustering = DBSCAN(eps=7, min_samples=1).fit(data)
        labels = clustering.labels_
        
        for label in set(labels):
            if label == -1: continue 
            
            cluster_indices = np.where(labels == label)[0]
            pts = [string_points[i] for i in cluster_indices]
            
            num_frames = len(set(p[3] for p in pts))
            raw_frets = [str(p[2]) for p in pts]
            
            processed_symbols = []
            harmonic_votes = 0
            ghost_votes = 0
            for f in raw_frets:
                if '<' in f and '>' in f and any(char.isdigit() for char in f): harmonic_votes += 1
                if '(' in f and ')' in f and any(char.isdigit() for char in f): ghost_votes += 1

                if label == -1: continue
                cluster_indices = np.where(labels == label)[0]
                pts = [string_points[i] for i in cluster_indices]

                num_frames = len(set(p[3] for p in pts))
                raw_frets = [str(p[2]) for p in pts]

                processed_symbols = []
                for f in raw_frets:
                    digit_match = "".join(filter(str.isdigit, f))
                    if digit_match: processed_symbols.append(digit_match)
                    elif 'X' in f.upper(): processed_symbols.append('X')
                    elif 'h' in f: processed_symbols.append('h')
                    elif 'p' in f: processed_symbols.append('p')
                    elif '/' in f: processed_symbols.append('/')
                    elif '\\' in f: processed_symbols.append('\\')
                    elif '|' in f: processed_symbols.append('|')
                    elif '$' in f: processed_symbols.append('$')
                
            # --- HEURISTIC: When to apply symbols ---
            is_harmonic = (harmonic_votes >= 1)
            is_ghost = (ghost_votes >= 1)

            # --- VALIDATION & CONSENSUS ---
            is_valid_note = num_frames > 1 or len(processed_symbols) > 0
            if not is_valid_note: continue

            avg_x = np.mean([p[0] for p in pts])
            
            if processed_symbols:
                consensus_val = max(set(processed_symbols), key=processed_symbols.count)
                if is_harmonic and consensus_val != 'X': consensus_val = f"<{consensus_val}>"
                if is_ghost and consensus_val != 'X': consensus_val = f"({consensus_val})"
            else:
                consensus_val = 'N'

            if consensus_val == 'N': continue # skip "N" values in the final tab
                
            final_tab.append({"x": avg_x, "string": s_idx, "fret": consensus_val}) 

    final_tab.sort(key=lambda n: n["x"])
    
    if DEBUG: show_stitching_debug(global_points, final_tab)
    save_final_tab(final_tab)
    return final_tab

def show_stitching_debug(global_points, final_tab):
    """
    Shows final tab results. Hovering over an area reveals ALL raw frame detections
    (including rejected noise) at that X-coordinate.
    """
    max_gx = int(max(p[0] for p in global_points)) if global_points else 1280
    view_offset = 0
    view_width = 1280
    mouse_x = -1
    
    def on_mouse(event, x, y, flags, param):
        nonlocal mouse_x
        mouse_x = x

    cv2.namedWindow("Interactive Stitcher")
    cv2.setMouseCallback("Interactive Stitcher", on_mouse)

    while True:
        canvas = np.ones((500, view_width, 3), dtype=np.uint8) * 255
        global_mouse_x = mouse_x + view_offset

        # 1. Background: Draw Strings
        for i in range(6):
            y_pos = 100 + (i * 50)
            cv2.line(canvas, (0, y_pos), (view_width, y_pos), (240, 240, 240), 1)

        # 2. Layer 1: The Final Stitched Tab (Always Visible)
        for note in final_tab:
            screen_x = int(note['x'] - view_offset)
            if 0 <= screen_x < view_width:
                y_pos = 100 + (note['string'] * 50)
                # Draw Final Note
                cv2.circle(canvas, (screen_x, y_pos), 15, (0, 165, 255), 2)
                cv2.putText(canvas, str(note['fret']), (screen_x - 6, y_pos + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 3. Layer 2: Hover Reveal (Raw Frame Notes)
        # Show raw data within 30 pixels of the mouse
        hover_radius = 30
        for gx, s_idx, fret, _ in global_points:
            dist = abs(gx - global_mouse_x)
            if dist < hover_radius:
                screen_x = int(gx - view_offset)
                if 0 <= screen_x < view_width:
                    y_pos = 100 + (s_idx * 50)
                    # Draw raw detection (Smaller, different color)
                    alpha = max(0.1, 1.0 - (dist / hover_radius)) # Fade based on distance
                    color = (150, 150, 150) if dist > 5 else (0, 0, 255) # Turn red if directly under mouse
                    cv2.circle(canvas, (screen_x, y_pos), 4, color, -1)
                    
                    # Show the fret value the OCR saw in that specific frame
                    cv2.putText(canvas, str(fret), (screen_x - 4, y_pos - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # 4. Vertical "Scanner" Line
        cv2.line(canvas, (mouse_x, 0), (mouse_x, 500), (200, 200, 200), 1)

        # UI
        cv2.putText(canvas, "A/D: Scroll | Hover to reveal raw frame data", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        cv2.imshow("Interactive Stitcher", canvas)
        
        key = cv2.waitKey(20) & 0xFF
        if key == ord('d'): view_offset += 100
        elif key == ord('a'): view_offset = max(0, view_offset - 100)
        elif key == ord(' ') or key == 13: break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit()

    cv2.destroyWindow("Interactive Stitcher")
    
def main():
    tab_data = import_raw_tab_data()
    offsets = read_list()
    final_tab = cluster_notes_to_tab(tab_data, offsets)

if __name__ == "__main__":
    main()