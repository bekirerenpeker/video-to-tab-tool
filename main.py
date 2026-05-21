import cv2

from calculate_offsets import handle_calculate_offsets
from note_reading import handle_note_reading
from stiching import handle_stitching
from tab_export import handle_export
from video_utils import handle_frames_fetching


def main():
    print("--- Guitar Tab Extractor CLI ---")

    handle_frames_fetching()
    handle_note_reading()
    handle_calculate_offsets()
    handle_stitching()
    handle_export()

    print("\nProcess complete!")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
