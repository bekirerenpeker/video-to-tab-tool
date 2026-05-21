import time

import cv2
import numpy as np

from data_export import import_raw_tab_data, save_list

DEBUG = True

INF, NEG_INF = 100000, -100000
MATCH_RADIUS = 10
JUMP_INTERVAL = 10

BARLINE_MATCH = 200
STROKE_MATCH = 150
ARTICULATION_MATCH = 80
NOTE_MATCH = 100
GENERIC_NOTE_MATCH = 40
FUZZY_MATCH = 20
STROKE_MISMATCH = 30
BARLINE_CONFLICT = -200
DIGIT_STRUCTURAL_CONFLICT = -150
DIGIT_MISMATCH = -100
DEFAULT_CONFLICT = -80
NO_MATCH_PENALTY = -150
UNMATCHED_PENALTY = -150

# TODO: refactor this code since the new note reading is better and has more features
# for example add different weights for bars, storkes, slides, hammer on, pull off etc.


def find_best_match_index(string1, string2, offset, note_idx):
    best_match_idx = -1
    best_match_score = 0  # 0 for different notes 1 for same note
    best_match_dist = INF

    for idx, note in enumerate(string2):
        dist = abs(note - note_idx)
        if dist > MATCH_RADIUS:
            continue

        score = 1 if string1[note_idx] == note else 0
        if score < best_match_score or (
            score == best_match_score and dist > best_match_dist
        ):
            continue

        best_match_idx = idx
        best_match_score = score
        best_match_dist = dist

    return best_match_idx, best_match_score, best_match_dist


def get_frame_bounds(frame):
    all_x = [note[0] for string in frame for note in string]
    if not all_x:
        return 0, 0
    return min(all_x), max(all_x)


def get_offset_range_between_frames(frame1, frame2):
    min1, max1 = get_frame_bounds(frame1)
    min2, max2 = get_frame_bounds(frame2)
    return 0, max1 - min2


def get_pair_score(val1, val2):
    v1, v2 = str(val1), str(val2)

    if v1 == v2:
        if v1 == "|":
            return BARLINE_MATCH
        if v1 in ("v", "^", "$"):
            return STROKE_MATCH
        if v1 in ("h", "p", "/", "\\", "_"):
            return ARTICULATION_MATCH
        if v1 == "N":
            return GENERIC_NOTE_MATCH
        if v1.isdigit():
            return NOTE_MATCH
        return ARTICULATION_MATCH

    if v1 == "N" or v2 == "N":
        return GENERIC_NOTE_MATCH

    fuzzy_set = {"h", "p", "/", "\\", "_"}
    if v1 in fuzzy_set and v2 in fuzzy_set:
        return FUZZY_MATCH

    if {v1, v2} <= {"v", "^"}:
        return STROKE_MISMATCH

    if v1 == "|" or v2 == "|":
        return BARLINE_CONFLICT

    is_v1_digit = v1.isdigit()
    is_v2_digit = v2.isdigit()
    if is_v1_digit != is_v2_digit:
        return DIGIT_STRUCTURAL_CONFLICT

    if is_v1_digit and is_v2_digit:
        return DIGIT_MISMATCH

    return DEFAULT_CONFLICT


def calculate_alignment_score(frame1, frame2, offset):
    frame1_min, frame1_max = get_frame_bounds(frame1)
    frame2_min, frame2_max = get_frame_bounds(frame2)

    total_score = 0
    total_dist = 0
    match_count = 0

    for s_idx in range(6):
        notes1 = frame1[s_idx]
        notes2 = frame2[s_idx]
        matched_in_f2 = set()

        for x1, val1 in notes1:
            target_x_in_f2 = x1 - offset
            is_visible_in_f2 = frame2_min <= target_x_in_f2 <= frame2_max

            best_match_idx = -1
            best_note_score = NEG_INF
            current_dist = INF

            for i2, (x2, val2) in enumerate(notes2):
                if i2 in matched_in_f2:
                    continue
                dist = abs(x2 - target_x_in_f2)
                if dist <= MATCH_RADIUS:
                    score = get_pair_score(val1, val2)

                    if score > best_note_score:
                        best_note_score, current_dist, best_match_idx = score, dist, i2

            if best_match_idx != -1:
                total_score += best_note_score
                total_dist += current_dist
                matched_in_f2.add(best_match_idx)
                match_count += 1
            elif is_visible_in_f2:
                total_score += UNMATCHED_PENALTY

        for i2, (x2, val2) in enumerate(notes2):
            if i2 not in matched_in_f2:
                original_x_in_f1 = x2 + offset
                if frame1_min <= original_x_in_f1 <= frame1_max:
                    total_score += UNMATCHED_PENALTY

    avg_dist = total_dist / max(1, match_count)
    return total_score, avg_dist


def calculate_alignment_score_in_range(
    frame1, frame2, min_offset, max_offset, interval
):
    offset = min_offset
    best_score, best_dist, best_offset = NEG_INF, INF, NEG_INF

    while offset < max_offset:
        score, dist = calculate_alignment_score(frame1, frame2, offset)
        print(f"offset = {offset} (score: {score}, dist: {dist}, interval: {interval})")
        if score > best_score or (score == best_score and dist < best_dist):
            best_score = score
            best_dist = dist
            best_offset = offset
        offset += interval

    return best_score, best_offset


def calculate_best_alignment(frame1, frame2):
    offset, max_offset = get_offset_range_between_frames(frame1, frame2)
    crude_score, crude_offset = calculate_alignment_score_in_range(
        frame1, frame2, offset, max_offset, JUMP_INTERVAL
    )
    if crude_offset == 0:
        return crude_score, crude_offset
    print(f"crude = {crude_offset} (score: {crude_score})")

    fine_score, fine_offset = calculate_alignment_score_in_range(
        frame1,
        frame2,
        max(crude_offset - JUMP_INTERVAL, 0),
        min(crude_offset + JUMP_INTERVAL, max_offset),
        1,
    )
    print(f"fine = {fine_offset} (score: {fine_score})")
    return fine_score, fine_offset


mouse_offset = 0


def on_mouse(event, x, y, flags, param):
    global mouse_offset
    # We use the x-coordinate of the mouse as a manual offset tester
    # Subtracting the 50px margin we use in the drawing
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_offset = max(0, x - 50)


def show_alignment_debug(frame1, frame2, best_offset, best_score, frame_idx):
    global mouse_offset
    cv2.namedWindow("Alignment Debugger")
    cv2.setMouseCallback("Alignment Debugger", on_mouse)

    # Initialize mouse_offset to the calculated best_offset for the first look
    mouse_offset = best_offset
    start_time = time.time()
    user_pressed_enter = False
    user_pressed_space = False
    frame1_min, frame1_max = get_frame_bounds(frame1)
    frame2_min, frame2_max = get_frame_bounds(frame2)
    min_offset, max_offset = get_offset_range_between_frames(frame1, frame2)

    while True:
        # Canvas: Height for 6 strings + space for info
        height, width = 600, 1400
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Draw string lines
        for i in range(6):
            y = 50 + (i * 50)
            cv2.line(canvas, (50, y), (width - 50, y), (200, 200, 200), 1)

        # 1. Draw Frame 1 (Red) - Static Reference
        for s_idx in range(6):
            y = 50 + (s_idx * 50)
            for x, val in frame1[s_idx]:
                cv2.circle(canvas, (int(x) + 50, y), 8, (0, 0, 255), 2)
                cv2.putText(
                    canvas,
                    str(val),
                    (int(x) + 45, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )

        # 2. Draw "Calculated Best" (Green) - Small static markers
        for s_idx in range(6):
            y = 50 + (s_idx * 50)
            for x, val in frame2[s_idx]:
                calc_x = int(x + best_offset) + 50
                if 50 <= calc_x <= width - 50:
                    cv2.circle(canvas, (calc_x, y), 12, (255, 0, 0), 1)
                    cv2.putText(
                        canvas,
                        str(val),
                        (calc_x - 5, y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                    )

        # 3. Draw "Ghost" Image (Blue) - Controlled by Mouse
        for s_idx in range(6):
            y = 50 + (s_idx * 50)
            for x, val in frame2[s_idx]:
                ghost_x = int(x + mouse_offset) + 50
                if 50 <= ghost_x <= width - 50:
                    cv2.drawMarker(
                        canvas,
                        (ghost_x, y),
                        (0, 150, 0),
                        cv2.MARKER_TILTED_CROSS,
                        10,
                        1,
                    )

        # 4. Draw Frame 1 bounds (Yellow)
        cv2.rectangle(
            canvas, (frame1_min + 50, 50), (frame1_max + 50, 350), (0, 255, 255), 2
        )
        cv2.putText(
            canvas,
            "FRAME 1",
            (frame1_min + 50, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

        # 5. Draw Frame 2 bounds (Blue) - Controlled by Mouse Offset
        f2_min_offsetted = int(frame2_min + mouse_offset) + 50
        f2_max_offsetted = int(frame2_max + mouse_offset) + 50
        cv2.rectangle(
            canvas, (f2_min_offsetted, 50), (f2_max_offsetted, 350), (255, 0, 0), 2
        )
        cv2.putText(
            canvas,
            "FRAME 2",
            (f2_min_offsetted, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

        # Info Text
        cv2.putText(
            canvas,
            f"FRAME {frame_idx} -> {frame_idx + 1}",
            (50, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            canvas,
            f"CALCULATED BEST OFFSET: {best_offset}",
            (50, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 150, 0),
            2,
        )
        cv2.putText(
            canvas,
            f"MOUSE (GHOST) OFFSET: {int(mouse_offset)}",
            (50, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )
        cv2.putText(
            canvas,
            f"OFFSET: {best_offset}",
            (50, 490),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            canvas,
            f"SCORE: {best_score}",
            (50, 520),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            canvas,
            f"OFFSET RANGE: {min_offset} - {max_offset}",
            (50, 550),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            canvas,
            f"FRAME 1 RANGE: {frame1_min} - {frame1_max}",
            (50, 580),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            canvas,
            f"FRAME 2 RANGE: {frame2_min} - {frame2_max}",
            (50, 610),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

        cv2.imshow("Alignment Debugger", canvas)

        key = cv2.waitKey(10) & 0xFF
        if key == 13:
            user_pressed_enter = True
            break
        elif key == ord(" "):
            user_pressed_space = not user_pressed_space
        elif key == ord("q"):
            cv2.destroyAllWindows()
            exit()
        if not user_pressed_space and (time.time() - start_time) > (
            0.5 if best_offset == 0 else 1.5
        ):
            break

    return not user_pressed_enter


def calculate_offsets(tab_data):
    offsets = []
    show_debug_window = DEBUG

    for i in range(len(tab_data) - 1):
        f1 = tab_data[i]
        f2 = tab_data[i + 1]
        best_score, best_offset = calculate_best_alignment(f1, f2)
        offsets.append(best_offset)
        print(f"Pair {i}-{i + 1}: Best Score {best_score} | Best Offset {best_offset}")
        if show_debug_window:
            show_debug_window = show_alignment_debug(f1, f2, best_offset, best_score, i)

    return offsets


def handle_calculate_offsets():
    print("\n==== Calculating offsets between frames... ====")
    tab_data = import_raw_tab_data()
    offsets = calculate_offsets(tab_data)
    save_list(offsets)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    handle_calculate_offsets()
