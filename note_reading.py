from export import read_list, export_raw_frames_visual, save_raw_tab_data
from note_positions import detect_notes
from ocr import debug_and_recognize_characters_threaded
from terminal_utils import draw_progress_bar
import os
import cv2

# doesn't go to the next frame until the user presses space
DEBUG=False

def read_notes(folder, string_y_positions):
    tab_data = []
    all_frames = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    total_frames = len(all_frames)

    for idx, f in enumerate(all_frames):
        img_path = os.path.join(folder, f)
        frame = cv2.imread(img_path)
        if frame is None: continue

        note_positions, arches, slides = detect_notes(frame, string_y_positions)
        
        notes, debug_frame = debug_and_recognize_characters_threaded(
            frame, 
            note_positions, 
            string_y_positions, 
            min_confidence=30
        )

        for s_idx in range(6):
            string_arches = arches[s_idx]
            string_slides = slides[s_idx]

            for center_x, bbox, orientation in string_arches:
                symbol = "h" if orientation == "up" else "p"
                notes[s_idx].append((center_x, symbol))

            for center_x, bbox, orientation in string_slides:
                symbol = "/" if orientation == "up" else "\\"
                notes[s_idx].append((center_x, symbol))

            notes[s_idx].sort(key=lambda n: n[0])
            
        tab_data.append(notes)
        draw_progress_bar(idx / total_frames, prefix=f"[{idx+1}/{total_frames}] Processed")
        
        # to see the results in real-time
        for i, sy_pos in enumerate(string_y_positions):
            for center_x, bbox, orientation in arches[i]:
                color = (0, 255, 0) if orientation == "up" else (255, 0, 0)
                cv2.circle(debug_frame, (center_x, sy_pos), 2, color, 2)
                cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 1)
            for center_x, bbox, orientation in slides[i]:
                color = (255, 0, 0) if orientation == "up" else (255, 255, 0)
                cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 1)

        cv2.imshow("Threaded Debug", debug_frame)

        if DEBUG:
            while True:
                key = cv2.waitKey(0) & 0xFF  # waitKey(0) waits indefinitely
                if key == ord(' '): break    # Space key to move to next frame
                if key == ord('q'):          # 'q' to quit the entire process
                    cv2.destroyAllWindows()
                    return tab_data
        else: 
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    save_raw_tab_data(tab_data)
    export_raw_frames_visual(tab_data)
    return tab_data

if __name__ == "__main__":
    string_y_positions = read_list(os.path.join("output", "string_positions.json"))
    read_notes(os.path.join("output", "frame_dump"), string_y_positions)
    cv2.destroyAllWindows()