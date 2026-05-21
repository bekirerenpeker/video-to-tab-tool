import json
import os

import cv2

DEBUG = False


def remove_single_template(
    processed_img, template_path, avg_spacing, target_ratio, threshold, mirror_vertical
):
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        return processed_img, []

    desired_h = int(avg_spacing * target_ratio)
    current_t_h, current_t_w = template.shape[:2]
    factor = desired_h / current_t_h

    # Resize template (using AREA for downscaling, CUBIC for upscaling)
    new_w = max(1, int(current_t_w * factor))
    interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
    resized_template = cv2.resize(template, (new_w, desired_h), interpolation=interp)

    # Perform Template Matching
    versions = [resized_template]
    if mirror_vertical:
        versions.append(cv2.flip(resized_template, 0))

    detected_locs = []
    for idx, t_ver in enumerate(versions):
        suffix = " (Mirrored)" if idx == 1 else ""
        while True:
            res = cv2.matchTemplate(processed_img, t_ver, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val < threshold:
                break

            # Record center X position
            center_x = max_loc[0] + new_w // 2
            detected_locs.append(center_x)

            cv2.rectangle(
                processed_img,
                max_loc,
                (max_loc[0] + new_w, max_loc[1] + desired_h),
                0,
                -1,
            )
            if DEBUG:
                print(f"Removed {os.path.basename(template_path)}{suffix} at {max_loc}")

    if DEBUG:
        display_img = processed_img.copy()
        img_h, img_w = display_img.shape[:2]
        margin = 15
        y1, y2 = margin, margin + desired_h
        x1, x2 = img_w - new_w - margin, img_w - margin

        if y2 < img_h and x1 > 0:
            display_img[y1:y2, x1:x2] = resized_template
            cv2.rectangle(display_img, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), 255, 1)
            cv2.putText(
                display_img,
                f"Ratio: {target_ratio}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                255,
                1,
            )

        cv2.imshow("Template Debug View", display_img)
        cv2.waitKey(0)

    return processed_img, detected_locs


def remove_all_templates(
    processed_img, avg_spacing, templates_path="templates/templates.json", threshold=0.6
):
    detected_templates = []
    if not os.path.exists(templates_path):
        return processed_img, []

    with open(templates_path, "r") as f:
        config = json.load(f)

    for entry in config.get("templates", []):
        if DEBUG:
            print(
                f"Processing template: {entry['path']} with {entry['target_ratio']} target ratio"
            )

        ascii_repr = entry.get("ascii", "------")
        if len(ascii_repr) != 6:
            ascii_repr = "------"

        processed_img, locs = remove_single_template(
            processed_img,
            entry["path"],
            avg_spacing,
            entry["target_ratio"],
            threshold=entry.get("threshold", threshold),
            mirror_vertical=entry.get("mirror_vertical", False),
        )

        for lx in locs:
            detected_templates.append((lx, ascii_repr))

    return processed_img, detected_templates
