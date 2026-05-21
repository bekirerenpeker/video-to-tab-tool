import os
import sys

import cv2
import yt_dlp

from data_export import save_list
from terminal_utils import draw_progress_bar

OUTPUT_DIR = "output"


def download_video(url, start_time, end_time):
    output_name = os.path.join(OUTPUT_DIR, "raw_download")

    ydl_opts = {
        # 1. 'noplaylist': True ensures it only grabs the single video
        "noplaylist": True,
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "outtmpl": f"{output_name}.%(ext)s",
        "download_sections": [
            {
                "start_time": start_time,
                "end_time": end_time,
            }
        ],
        "force_keyframes_at_cuts": True,
        # 2. Add preference for mp4 to make OpenCV's life easier
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Return the expected filename
    return f"{output_name}.mp4"


def extract_frames(video_path, start_seconds, end_seconds, interval):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * interval)

    # Calculate start and end frame indices
    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps)
    total_to_process = end_frame - start_frame

    # 1. ROI Selection (Using start frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + int(0.5 * fps))
    ret, selection_frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, selection_frame = cap.read()

    print("\n[ACTION] Select the TAB area and press ENTER.")
    roi = cv2.selectROI("Select Tabs", selection_frame, False)
    cv2.destroyWindow("Select Tabs")
    x, y, w, h = roi

    output_folder = os.path.join(OUTPUT_DIR, "frame_dump")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. Reset to start_frame for actual extraction
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    saved = 0

    print(f"\nExtracting from {start_seconds}s to {end_seconds}s...")

    while current_frame <= end_frame:
        ret = cap.grab()
        if not ret:
            break

        relative_frame = current_frame - start_frame
        if relative_frame % interval_frames == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break
            crop = frame[y : y + h, x : x + w]
            cv2.imwrite(f"{output_folder}/frame_{saved:04d}.png", crop)
            saved += 1

            # Update progress bar relative to the section, not the whole video
            progress = min(
                1.0, (current_frame - start_frame) / max(1, total_to_process)
            )
            draw_progress_bar(progress, prefix="Extracting")

        current_frame += 1

    cap.release()
    print(f"\nSaved {saved} frames.")
    return output_folder


def calibrate_strings(sample_frame):
    print("\n[ACTION] Click on each of the 6 strings (Top to Bottom).")
    y_coords = []

    def click_event(event, x, y, flags, param):
        display_img = sample_frame.copy()
        # Horizontal Preview Line
        cv2.line(display_img, (0, y), (sample_frame.shape[1], y), (255, 255, 0), 1)

        # Draw existing points
        for yc in y_coords:
            cv2.circle(display_img, (x, yc), 3, (0, 0, 255), -1)

        cv2.imshow("Calibrate Strings", display_img)

        if event == cv2.EVENT_LBUTTONDOWN:
            y_coords.append(y)
            print(f"String {len(y_coords)} set at Y={y}")

    cv2.imshow("Calibrate Strings", sample_frame)
    cv2.setMouseCallback("Calibrate Strings", click_event)

    while len(y_coords) < 6:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyWindow("Calibrate Strings")
    return sorted(y_coords)


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


def handle_frames_fetching():
    url = input("YouTube URL: ")
    start_seconds = convert_to_seconds(input("Start Time (default 0.02s): ") or "0.02")
    end_seconds = convert_to_seconds(input("End Time (default 1.00s): ") or "1.00")
    interval = float(input("Frame interval (default 1.0): ") or 1.0)

    print(f"\n==== Downloading section {start_seconds}s to {end_seconds}s... ====")
    try:
        video_file = download_video(url, start_seconds, end_seconds)
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    print("\n==== Opening video for ROI selection... ====")
    folder = extract_frames(video_file, start_seconds, end_seconds, interval)
    if folder:
        print(f"\nSuccess! Frames are stored in: {folder}")
    else:
        print("\nExtraction failed.")
        sys.exit(1)

    print("\n==== Calibrating string positions... ====")
    sample_img_path = os.path.join(folder, "frame_0000.png")
    sample_img = cv2.imread(sample_img_path)
    string_y_positions = calibrate_strings(sample_img)
    save_list(string_y_positions, os.path.join("output", "string_positions.json"))
