import os

from data_export import load_final_tab


def export_raw_frames_visual(
    all_frame_results, filename=os.path.join("output", "raw_frames_visual.txt")
):
    """Exports each frame as its own individual ASCII tab block."""
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
                    frame_notes.append({"x": x, "s": s_idx, "f": str(fret)})

            # Sort notes by x position within the frame
            frame_notes.sort(key=lambda n: n["x"])

            # Build the columns based on X position
            # We use a small tolerance (e.g., 10px) to group simultaneous notes
            columns = []
            if frame_notes:
                current_col = [frame_notes[0]]
                for i in range(1, len(frame_notes)):
                    if frame_notes[i]["x"] - current_col[-1]["x"] <= 10:
                        current_col.append(frame_notes[i])
                    else:
                        columns.append(current_col)
                        current_col = [frame_notes[i]]
                columns.append(current_col)

            # Map columns to the string lines
            for col in columns:
                notes_in_col = {n["s"]: n["f"] for n in col}
                max_w = max(len(n["f"]) for n in col)

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
    max_line_length=200,
    spacing_ratio=0.12,
    padding=1,
):
    """
    Exports final tab data with spatial spacing and side padding.

    final_tab: List of dicts [{'x': float, 'string': int, 'fret': str}, ...]
    max_line_length: Maximum characters per block.
    spacing_ratio: Ratio of X-distance to dashes.
    padding: Number of dashes to add after the start pipe and before the end pipe.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 1. Sort by X
    final_tab.sort(key=lambda n: n["x"])

    columns = []
    if final_tab:
        current_col = [final_tab[0]]
        # Keep track of which strings are already used in the current column
        used_strings = {final_tab[0]["string"]}

        for i in range(1, len(final_tab)):
            note = final_tab[i]
            x_dist = note["x"] - current_col[-1]["x"]

            # CONDITION: Merge into column ONLY if:
            # 1. The X distance is very small (chord)
            # 2. AND the string isn't already taken (no horizontal overlap)
            if x_dist <= 8 and note["string"] not in used_strings:
                current_col.append(note)
                used_strings.add(note["string"])
            else:
                # Force a new column
                columns.append(current_col)
                current_col = [note]
                used_strings = {note["string"]}
        columns.append(current_col)

    # 2. Build the tab buffers
    pad_str = "-" * padding
    full_strings = [["|" + pad_str] for _ in range(6)]

    # Start last_x from the first note's position minus a small buffer
    # so the first note doesn't have a massive gap from the pipe
    last_x = columns[0][0]["x"] if columns else 0

    for col in columns:
        current_x = col[0]["x"]
        dist = current_x - last_x

        # Calculate dashes based on distance, but ensure at least 1
        # dash between separate columns so they don't touch
        num_dashes = max(1, int(dist * spacing_ratio))

        for s_idx in range(6):
            full_strings[s_idx].append("-" * num_dashes)

        # Determine the width needed for this column (e.g., "12" needs 2 chars)
        max_fret_w = max(len(str(n["fret"])) for n in col)
        notes_in_col = {n["string"]: str(n["fret"]) for n in col}

        for s_idx in range(6):
            fret_val = notes_in_col.get(s_idx, "-")
            # Fill with dashes to keep all strings aligned
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

            # Look for a barline to break on if it's close to the end of the line (e.g. >= 70% of effective_max)
            best_barline_end = None
            temp_width = 0
            for idx in range(curr_seg_idx, line_end):
                temp_width += len(full_strings[0][idx])

                # Check if this segment is a barline fret/symbol column
                is_bar = False
                if idx > 0 and idx % 2 == 0:
                    col_idx = (idx - 2) // 2
                    if col_idx < len(columns):
                        is_bar = any(str(n["fret"]) == "|" for n in columns[col_idx])

                if is_bar:
                    if temp_width >= 0.7 * effective_max:
                        best_barline_end = (
                            idx + 1
                        )  # break right after this barline segment

            if best_barline_end is not None:
                line_end = best_barline_end

            if line_end == curr_seg_idx:
                line_end += 1

            # Check if this line is ending exactly on a barline segment
            ends_on_barline = False
            if line_end > 0 and (line_end - 1) % 2 == 0:
                col_idx = (line_end - 1 - 2) // 2
                if col_idx < len(columns):
                    ends_on_barline = any(
                        str(n["fret"]) == "|" for n in columns[col_idx]
                    )

            for s_idx in range(6):
                line_text = "".join(full_strings[s_idx][curr_seg_idx:line_end])

                # If the line ends on a barline, it already has the closing pipe '|'.
                # Otherwise, we need to add the right padding and close the pipe.
                if ends_on_barline:
                    f.write(line_text + "\n")
                else:
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


def handle_export():
    print("\n==== Exporting to Tablature... ====")
    final_tab = load_final_tab()
    export_stitched_tab_visual(final_tab, spacing_ratio=0.08)


if __name__ == "__main__":
    handle_export()
