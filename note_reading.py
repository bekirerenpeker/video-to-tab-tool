import re
from export import read_list, export_raw_frames_visual, save_raw_tab_data
from note_positions import detect_notes
from ocr import debug_and_recognize_characters_threaded
from terminal_utils import draw_progress_bar
import os
import cv2

# doesn't go to the next frame until the user presses space
DEBUG=False

def merge_notes_and_articulations(avg_spacing, notes, arches, slides, bars, arp_strokes):
    merged_notes = [sorted(string_notes, key=lambda n: n[0]) for string_notes in notes]

    for s_idx in range(6):
        string_arches = arches[s_idx]
        string_slides = slides[s_idx]

        for center_x, (x,y,w,h), orientation in string_arches:
            starting_note = next((n for n in merged_notes[s_idx][::-1] if n[0] < x+5), None)
            ending_note = next((n for n in merged_notes[s_idx] if n[0] > x+w-5), None)
            notes_inbetween = [n for n in merged_notes[s_idx] if x+5 < n[0] < x+w-5]
            effected_notes = [starting_note] + notes_inbetween + [ending_note]
            if not starting_note or not ending_note or len(effected_notes) < 2: continue

            if len(notes_inbetween) == 0 and w > avg_spacing*2:
                merged_notes[s_idx].append((center_x, "_"))
                continue

            for note1, note2 in zip(effected_notes[:-1], effected_notes[1:]):
                note1_clean = re.sub(r'\D', '', note1[1])
                note2_clean = re.sub(r'\D', '', note2[1])
                
                digit1 = int(note1_clean) if note1_clean.isdigit() else 0 
                digit2 = int(note2_clean) if note2_clean.isdigit() else 0 
                
                symbol = "h" if digit1 < digit2 else "p"
                merged_notes[s_idx].append(((note1[0] + note2[0])//2, symbol))

        for center_x, bbox, orientation in string_slides:
            symbol = "/" if orientation == "up" else "\\"
            merged_notes[s_idx].append((center_x, symbol))
        
    for x_pos, s_idx, e_idx in arp_strokes:
        for i in range(s_idx, e_idx + 1):
            merged_notes[i].append((x_pos, "$"))

    for bar_x_pos in bars:
        for i in range(6):
            merged_notes[i].append((bar_x_pos, "|"))

    return [sorted(string_notes, key=lambda n: n[0]) for string_notes in merged_notes]

def read_notes(folder, string_y_positions):
    tab_data = []
    all_frames = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    total_frames = len(all_frames)

    for idx, f in enumerate(all_frames):
        img_path = os.path.join(folder, f)
        frame = cv2.imread(img_path)
        if frame is None: continue

        avg_spacing = abs(string_y_positions[0] - string_y_positions[-1]) / 5
        note_positions, arches, slides, bars, arp_strokes = detect_notes(frame, string_y_positions)
        
        notes, debug_frame = debug_and_recognize_characters_threaded(
            frame, 
            note_positions, 
            string_y_positions, 
            min_confidence=30
        )

        notes = merge_notes_and_articulations(avg_spacing, notes, arches, slides, bars, arp_strokes)
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
            for bar_x_pos in bars:
                cv2.line(debug_frame, (bar_x_pos, string_y_positions[0]), (bar_x_pos, string_y_positions[5]), (255, 0, 255), 1)
            for x_pos, s_idx, e_idx in arp_strokes:
                cv2.line(debug_frame, (x_pos, string_y_positions[s_idx]), (x_pos, string_y_positions[e_idx]), (255, 255, 0), 2)

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