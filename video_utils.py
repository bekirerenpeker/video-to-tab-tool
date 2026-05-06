from terminal_utils import draw_progress_bar
import cv2
import os
import yt_dlp

OUTPUT_DIR = "output"

def download_video(url, start_time, end_time):
    output_name = os.path.join(OUTPUT_DIR, "raw_download")

    ydl_opts = {
        # 1. 'noplaylist': True ensures it only grabs the single video
        'noplaylist': True,
        'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
        'outtmpl': f'{output_name}.%(ext)s',
        'download_sections': [{
            'start_time': start_time,
            'end_time': end_time,
        }],
        'force_keyframes_at_cuts': True,
        # 2. Add preference for mp4 to make OpenCV's life easier
        'merge_output_format': 'mp4',
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
    if not os.path.exists(output_folder): os.makedirs(output_folder)


    # 2. Reset to start_frame for actual extraction
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    saved = 0
    
    print(f"\nExtracting from {start_seconds}s to {end_seconds}s...")

    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret: break
        
        # Only save if we are at the interval step
        relative_frame = current_frame - start_frame
        if relative_frame % interval_frames == 0:
            crop = frame[y:y+h, x:x+w]
            cv2.imwrite(f"{output_folder}/frame_{saved:04d}.png", crop)
            saved += 1
        
        # Update progress bar relative to the section, not the whole video
        progress = min(1.0, (current_frame - start_frame) / max(1, total_to_process))
        draw_progress_bar(progress, prefix='Extracting')
        
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
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cv2.destroyWindow("Calibrate Strings")
    return sorted(y_coords)