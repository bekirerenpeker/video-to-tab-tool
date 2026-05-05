from note_positions import merge_close_points
from export import read_offset_values
from export import export_raw_frames_visual, save_raw_tab_data
from note_positions import detect_note_bboxes
from ocr import debug_and_recognize_characters_threaded
from terminal_utils import draw_progress_bar
import os
import cv2

def read_notes(folder, string_y_positions, debug=True, wait_for_space=False):
    tab_data = []
    all_frames = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    total_frames = len(all_frames)

    for idx, f in enumerate(all_frames):
        img_path = os.path.join(folder, f)
        frame = cv2.imread(img_path)
        if frame is None: continue

        raw_notes_data = detect_note_bboxes(frame, string_y_positions, debug=debug)
        merged_notes_data = merge_close_points(raw_notes_data, min_dist=7)
        notes, debug_frame = debug_and_recognize_characters_threaded(
            frame, 
            merged_notes_data, 
            string_y_positions, 
            min_confidence=30
        )
        tab_data.append(notes)

        draw_progress_bar(idx / total_frames, prefix=f"[{idx+1}/{total_frames}] Processed")
        
        # to see the results in real-time
        cv2.imshow("Threaded Debug", debug_frame)

        if wait_for_space:
            print(f"\n[PAUSED] Frame {idx}. Press SPACE to continue, 'q' to quit...")
            while True:
                key = cv2.waitKey(0) & 0xFF  # waitKey(0) waits indefinitely
                if key == ord(' '):      # Space key to move to next frame
                    break
                if key == ord('q'):      # 'q' to quit the entire process
                    cv2.destroyAllWindows()
                    return tab_data
        else:
            # Normal real-time viewing mode
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

    save_raw_tab_data(tab_data)
    export_raw_frames_visual(tab_data)
    return tab_data

if __name__ == "__main__":
    string_y_positions = read_offset_values(os.path.join("output", "string_positions.json"))
    read_notes(os.path.join("output", "frame_dump"), string_y_positions, debug=True, wait_for_space=False)
    cv2.destroyAllWindows()