import json, os, shutil

def cleanup_previous_data(output_dir="output"):
    """Removes the old frames and video files if they exist."""
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

def import_raw_tab_data(filename=os.path.join("output", "tab_data.json")):
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

def save_list(values, filename=os.path.join("output", "offsets.json")):
    """Saves the list of integers to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(values, f, indent=4)
    print(f"List saved to {filename}")

def read_list(filename=os.path.join("output", "offsets.json")):
    """Reads the list of integers from a JSON file."""
    if not os.path.exists(filename):
        print(f"No file found at {filename}")
        return None
    with open(filename, "r") as f: data = json.load(f)
    return data

def save_final_tab(final_tab_data, filename=os.path.join("output", "final_tab.json")):
    """ Saves the post-clustering/stitched tab data.  """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f: json.dump(final_tab_data, f, indent=4)
    print(f"Final tab saved to {filename}")

def load_final_tab(filename=os.path.join("output", "final_tab.json")):
    """ Loads the stitched tab data back into a list of dictionaries.  """
    if not os.path.exists(filename): return None
    with open(filename, "r") as f: return json.load(f)
