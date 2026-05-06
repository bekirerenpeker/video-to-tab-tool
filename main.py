from export import save_list
from note_reading import read_notes
from export import cleanup_previous_data, export_stitched_tab_visual
from video_utils import download_video, extract_frames, calibrate_strings
from calculate_offsets import calculate_offsets
from stiching import cluster_notes_to_tab
import sys
import os
import cv2

def convert_to_seconds(time_str):
    """
    Converts M.SS or MM.SS format into total integer seconds.
    Example: 1.23 -> 83
    """
    try:
        if "." in time_str:
            minutes, seconds = time_str.split(".")
            # Ensure seconds part is treated as 2 digits (e.g., .5 is .05)
            total_seconds = int(minutes) * 60 + int(seconds)
            return total_seconds
        else:
            return int(time_str)
    except ValueError:
        print(f"Invalid time format: {time_str}. Use M.SS (e.g. 1.23)")
        sys.exit(1)

def main():
    print("--- Guitar Tab Extractor CLI ---")

    print("Cleaning up previous data...")
    cleanup_previous_data()
    
    url = input("YouTube URL: ")
    start_raw = input("Start Time (M.SS, e.g., 1.23): ")
    end_raw = input("End Time (M.SS, e.g., 2.45): ")
    start_seconds = convert_to_seconds(start_raw or "0.02")
    end_seconds = convert_to_seconds(end_raw or "1.00")
    interval = float(input("Frame interval (e.g., 1.0 for every second): ") or 1.0)

    print(f"\n==== Downloading section {start_seconds}s to {end_seconds}s... ====")
    try: video_file = download_video(url, start_seconds, end_seconds)
    except Exception as e: print(f"Download failed: {e}"); sys.exit(1)

    print(f"\n==== Opening video for ROI selection... ====")
    folder = extract_frames(video_file, start_seconds, end_seconds, interval)
    if folder: print(f"\nSuccess! Frames are stored in: {folder}")
    else: print("\nExtraction failed."); sys.exit(1)

    print("\n==== Calibrating string positions... ====")
    sample_img_path = os.path.join(folder, "frame_0000.png")
    sample_img = cv2.imread(sample_img_path)
    string_y_positions = calibrate_strings(sample_img)
    save_list(string_y_positions, os.path.join("output", "string_positions.json"))

    print("\n==== Processing frames into data... ====")
    tab_data = read_notes(folder, string_y_positions)

    print("\n==== Calculating offsets between frames... ====")
    offsets = calculate_offsets(tab_data)

    print("\n==== Stitching frames together... ====")
    final_tab = cluster_notes_to_tab(tab_data, offsets)

    print("\n==== Exporting to Tablature... ====")
    export_stitched_tab_visual(final_tab)

    print("\nProcess complete!")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()