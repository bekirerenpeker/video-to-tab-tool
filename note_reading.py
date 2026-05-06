from export import read_list, export_raw_frames_visual, save_raw_tab_data
from note_positions import detect_notes
from ocr import debug_and_recognize_characters_threaded
from terminal_utils import draw_progress_bar
import os
import cv2

DEBUG=True

def read_notes(folder, string_y_positions):
    tab_data = []
    all_frames = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    total_frames = len(all_frames)

    for idx, f in enumerate(all_frames):
        img_path = os.path.join(folder, f)
        frame = cv2.imread(img_path)
        if frame is None: continue

        note_positions = detect_notes(frame, string_y_positions)
        
        notes, debug_frame = debug_and_recognize_characters_threaded(
            frame, 
            note_positions, 
            string_y_positions, 
            min_confidence=30
        )
        tab_data.append(notes)

        draw_progress_bar(idx / total_frames, prefix=f"[{idx+1}/{total_frames}] Processed")
        
        # to see the results in real-time
        cv2.imshow("Threaded Debug", debug_frame)

        if DEBUG:
            while True:
                key = cv2.waitKey(0) & 0xFF  # waitKey(0) waits indefinitely
                if key == ord(' '): break # Space key to move to next frame
                if key == ord('q'):       # 'q' to quit the entire process
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