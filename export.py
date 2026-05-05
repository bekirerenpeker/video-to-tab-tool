import json, os, shutil
import xml.etree.ElementTree as ET

def cleanup_previous_data(output_dir="output"):
    """Removes the old frames and video files if they exist."""
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

def import_tab_data(filename=os.path.join("output", "tab_data.json")):
    if not os.path.exists(filename): return None
    with open(filename, "r") as f:
        data = json.load(f)
    
    all_frames = []
    for entry in data:
        frame = [[] for _ in range(6)]
        for n in entry["notes"]:
            frame[n["string"]].append((n["x_pos"], n["fret"]))
        all_frames.append(frame)
    return all_frames

def save_raw_tab_data(all_frame_results, filename=os.path.join("output", "tab_data.json")):
    structured = []
    for i, frame in enumerate(all_frame_results):
        notes = []
        for s_idx, string in enumerate(frame):
            for x, fret in string:
                notes.append({"string": s_idx, "fret": fret, "x_pos": x})
        
        notes.sort(key=lambda n: n["x_pos"])
        structured.append({"frame": i, "notes": notes})

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f: json.dump(structured, f, indent=4)

def save_offset_values(offsets, filename=os.path.join("output", "offsets.json")):
    """Saves the list of integer offsets to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump({"offsets": offsets}, f, indent=4)
    print(f"Offsets saved to {filename}")

def read_offset_values(filename=os.path.join("output", "offsets.json")):
    """Reads the list of integer offsets from a JSON file."""
    if not os.path.exists(filename):
        print(f"No offset file found at {filename}")
        return None
    with open(filename, "r") as f: data = json.load(f)
    return data.get("offsets", [])

def save_final_tab(final_tab_data, filename=os.path.join("output", "final_tab.json")):
    """
    Saves the post-clustering/stitched tab data.
    final_tab_data: List of dicts [{"x": 123, "string": 0, "fret": "7"}, ...]
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(final_tab_data, f, indent=4)
    print(f"Final tab saved to {filename}")

def load_final_tab(filename=os.path.join("output", "final_tab.json")):
    """
    Loads the stitched tab data back into a list of dictionaries.
    """
    if not os.path.exists(filename):
        return None
    with open(filename, "r") as f:
        return json.load(f)

def export_raw_frames_visual(all_frame_results, filename=os.path.join("output", "raw_frames_visual.txt")):
    """
    Exports each frame as its own individual ASCII tab block.
    all_frame_results: [ [[(x, fret),...], [string1], ...], [frame2...]]
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, "w") as f:
        for frame_idx, frame_data in enumerate(all_frame_results):
            # Header for the frame
            f.write(f"--- FRAME {frame_idx:04d} ---\n")
            
            # Create a 6-string display for this specific frame
            # String 0 is High E, String 5 is Low E
            tab_lines = [["|"] for _ in range(6)]
            
            # Get all notes in this frame to find horizontal order
            frame_notes = []
            for s_idx, string_notes in enumerate(frame_data):
                for x, fret in string_notes:
                    frame_notes.append({'x': x, 's': s_idx, 'f': str(fret)})
            
            # Sort notes by x position within the frame
            frame_notes.sort(key=lambda n: n['x'])
            
            # Build the columns based on X position
            # We use a small tolerance (e.g., 10px) to group simultaneous notes
            columns = []
            if frame_notes:
                current_col = [frame_notes[0]]
                for i in range(1, len(frame_notes)):
                    if frame_notes[i]['x'] - current_col[-1]['x'] <= 10:
                        current_col.append(frame_notes[i])
                    else:
                        columns.append(current_col)
                        current_col = [frame_notes[i]]
                columns.append(current_col)

            # Map columns to the string lines
            for col in columns:
                notes_in_col = {n['s']: n['f'] for n in col}
                max_w = max(len(n['f']) for n in col)
                
                for s_idx in range(6):
                    char = notes_in_col.get(s_idx, "-")
                    tab_lines[s_idx].append(char.ljust(max_w, "-") + "-")

            # Write the strings to the file
            for line in tab_lines:
                f.write("".join(line) + "|\n")
            
            # Add spacing between frames
            f.write("\n" + " " * 20 + "\n\n")

    print(f"Raw visual frames saved to {filename}")

def export_stitched_tab_visual(
    final_tab, 
    filename=os.path.join("output", "stitched_tab.txt"), 
    max_line_length=180, 
    spacing_ratio=0.1,
    padding=3
):
    """
    Exports final tab data with spatial spacing and side padding.
    
    final_tab: List of dicts [{'x': float, 'string': int, 'fret': str}, ...]
    max_line_length: Maximum characters per block.
    spacing_ratio: Ratio of X-distance to dashes.
    padding: Number of dashes to add after the start pipe and before the end pipe.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 1. Sort and group notes into columns (chords) by X coordinate
    final_tab.sort(key=lambda n: n['x'])
    columns = []
    if final_tab:
        current_col = [final_tab[0]]
        for i in range(1, len(final_tab)):
            if final_tab[i]['x'] - current_col[-1]['x'] <= 10:
                current_col.append(final_tab[i])
            else:
                columns.append(current_col)
                current_col = [final_tab[i]]
        columns.append(current_col)

    # 2. Build the tab buffers with spacing and internal padding
    # Each string starts with a pipe and the left padding
    pad_str = "-" * padding
    full_strings = [["|" + pad_str] for _ in range(6)]
    last_x = columns[0][0]['x'] if columns else 0
    
    for col in columns:
        current_x = col[0]['x']
        # Calculate horizontal distance[cite: 1]
        dist = current_x - last_x
        num_dashes = max(1, int(dist * spacing_ratio))
        
        for s_idx in range(6):
            full_strings[s_idx].append("-" * num_dashes)
        
        # Place the notes/chords (keeping 'N' values per request)
        notes_in_col = {n['string']: str(n['fret']) for n in col}
        max_fret_w = max(len(str(n['fret'])) for n in col)
        
        for s_idx in range(6):
            fret_val = notes_in_col.get(s_idx, "-")
            full_strings[s_idx].append(fret_val.ljust(max_fret_w, "-"))
            
        last_x = current_x

    # 3. Write with block-wrapping and right padding
    with open(filename, "w") as f:
        f.write("VIDEO-TO-TAB AI: FINAL STICHED TAB\n")
        f.write("=" * max_line_length + "\n\n")
        
        total_segments = len(full_strings[0])
        curr_seg_idx = 0
        
        while curr_seg_idx < total_segments:
            line_end = curr_seg_idx
            accumulated_width = 0
            
            # Account for the right padding and pipe in width calculation
            # (effective_max = total_max - padding - 1)
            effective_max = max_line_length - padding - 1
            
            while line_end < total_segments:
                segment_len = len(full_strings[0][line_end])
                if accumulated_width + segment_len > effective_max:
                    break
                accumulated_width += segment_len
                line_end += 1
            
            if line_end == curr_seg_idx:
                line_end += 1

            for s_idx in range(6):
                line_text = "".join(full_strings[s_idx][curr_seg_idx:line_end])
                
                # If this is the start of a block, it already has | and left padding
                # We just need to add the right padding and close the pipe
                f.write(line_text + pad_str + "|\n")
            
            f.write("\n")
            
            # For subsequent blocks, ensure they start with a pipe and padding
            curr_seg_idx = line_end
            if curr_seg_idx < total_segments:
                # Replace the next "start" segment if it doesn't have a pipe
                if not str(full_strings[0][curr_seg_idx]).startswith("|"):
                    for s_idx in range(6):
                        full_strings[s_idx][curr_seg_idx] = "|" + pad_str
    
    print(f"Visual tab saved to {filename}")

if __name__ == "__main__":
    final_tab = load_final_tab()
    export_stitched_tab_visual(final_tab, spacing_ratio=0.08)