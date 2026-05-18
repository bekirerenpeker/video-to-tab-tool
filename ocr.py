import random
import cv2
import pytesseract
import numpy as np
import concurrent.futures
from queue import Queue

# NOTE: Set your Tesseract path before initializing the pool if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
SKIP_OCR=False

# TODO: optimize the ocr so that it runs faster it is curretly the bottleneck of the program
LOOKALIKES = {
    "0": ["O","o","Q", "U", "e)", "ce"],
    "1": ["I", "l", "41"],
    "2": ["Z", "z", "22"],
    "5": ["?"],
    "6": ["b"],
    "7": ["C", "t", "A"],
    "8": ["B", "&"],
    "X": ["x"],
}

class TesseractThreadPool:
    """
    Manages a pool of individual Tesseract engine instances to allow 
    safe multithreaded character recognition on multiple ROIs.
    """
    def __init__(self, pool_size=4):
        # PSM 13: Raw line mode. Better for 1-2 digits than PSM 10 (Single Char).
        # PSM 7 (Single line) is also a strong alternative if 13 struggles.
        self.config = r'--psm 13'
        self.engine_pool = Queue()
        self.pool_size = pool_size
        for _ in range(pool_size): self.engine_pool.put(self.config)

    def process_roi(self, roi):
        if SKIP_OCR: return "X", 100

        """Worker thread function: borrows an engine and OCRs the ROI."""
        engine_config = self.engine_pool.get()
        best_char = ""
        best_conf = -1

        try:
            # Use image_to_data to get per-character or per-word confidence
            data = pytesseract.image_to_data(roi, config=engine_config, output_type=pytesseract.Output.DICT)
            
            # Filter and join text results
            text_segments = []
            confidences = []

            for j in range(len(data['text'])):
                text = data['text'][j].strip()
                if text:
                    text_segments.append(text)
                    confidences.append(int(data['conf'][j]))
            
            if text_segments:
                raw_text = "".join(text_segments)
                best_conf = int(np.mean(confidences))

                # 1. First, check if the ENTIRE raw_text is a multi-char misread (like '41')
                found_full_match = False
                for target, misreads in LOOKALIKES.items():
                    if raw_text in misreads:
                        best_char = target
                        found_full_match = True
                        break
                        
                # 2. If no full match, check character by character
                if not found_full_match:
                    mapped_chars = []
                    for char in raw_text:
                        found_char_match = False
                        for target, misreads in LOOKALIKES.items():
                            if char in misreads:
                                mapped_chars.append(target)
                                found_char_match = True
                                break
                        
                        if not found_char_match:
                            mapped_chars.append(char)
                    
                    best_char = "".join(mapped_chars)

                # Special case: reduce sequences of 0s if read as '00'
                if best_char.isnumeric() and len(best_char) > 1:
                    try: best_char = str(int(best_char))
                    except ValueError: pass
        finally:
            # 3. CRUCIAL: Always return the engine to the pool, even if OCR fails
            self.engine_pool.put(engine_config)

        return best_char, best_conf

# Initialize the pool ONCE outside the main loop
# Set pool_size based on your CPU cores (usually 4-8 is optimal for Tesseract)
_ocr_pool = TesseractThreadPool(pool_size=4)

def debug_and_recognize_characters_threaded(frame, all_notes_data, string_y_positions, min_confidence=40):
    """
    Multithreaded version of character recognition. Processes ROIs concurrently.
    Maintain same pre-processing and 'Trust but Verify' logic as single-threaded.
    """
    debug_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    padding_px = 10
    pre_processed_rois = []
    roi_metadata = []

    for i, y_pos in enumerate(string_y_positions):
        for x_pos, [x, y, w, h] in all_notes_data[i]:
            roi = cv2.getRectSubPix(gray, (w, h), (x+w//2, y+h//2))

            if roi is not None:
                # Ensure black text on white background (Crucial for Tesseract)
                edge_mask = np.ones(roi.shape, dtype=bool)
                edge_mask[1:-1, 1:-1] = False
                if np.mean(roi[edge_mask]) < 127: 
                    roi = cv2.bitwise_not(roi)

                # STRING STRIPPER (Fixes 1 being read as 4 or 7)
                binary_inv = cv2.bitwise_not(roi)
                string_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
                detected_lines = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, string_kernel)
                roi = cv2.subtract(binary_inv, detected_lines)
                roi = cv2.bitwise_not(roi) # Invert back to black text on white

                # GAUSSIAN BLUR (Fixes 0 being read as 6)
                roi = cv2.GaussianBlur(roi, (3, 3), 0)

                # Resize to a height of about 30-50 pixels (Tesseract's sweet spot)
                scaling = 45.0 / roi.shape[0]
                roi = cv2.resize(roi, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_CUBIC)
                
                # Clean threshold
                _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Standardized Padding (The 'Moat')
                roi = cv2.copyMakeBorder(roi, padding_px, padding_px, padding_px, padding_px, 
                                        cv2.BORDER_CONSTANT, value=[255, 255, 255])
                
                pre_processed_rois.append(roi)
                roi_metadata.append({
                    'string_idx': i,
                    'x': x_pos,
                    'bbox': (int(x), int(y), int(x + w), int(y + h))
                })

    results = [[] for _ in range(6)]
    if not pre_processed_rois: return results, debug_frame

    # Concurrent Character Recognition
    # Submit all pre-processed ROIs to the thread pool for simultaneous processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=_ocr_pool.pool_size) as executor:
        # map ensures that ocr_results are returned in the exact same order as pre_processed_rois
        ocr_results = list(executor.map(_ocr_pool.process_roi, pre_processed_rois))

    # Assemble Results and Draw Debug
    # Because 'map' preserved the order, we can zip ocr_results and roi_metadata together
    for metadata, ocr_data in zip(roi_metadata, ocr_results):
        char, conf = ocr_data
        string_i = metadata['string_idx']
        x_p = metadata['x']
        x1, y1, x2, y2 = metadata['bbox']

        # 'Trust but Verify' Logic:
        # Accept if confident enough, OR if Tesseract found anything in our restricted whitelist.
        is_confident = (conf >= min_confidence) or (char != "")
        color = (0, 255, 0) if is_confident else (0, 0, 255)

        if is_confident and char:
            results[string_i].append((x_p, char))
        else:
            results[string_i].append((x_p, "N"))

        color = (random.randint(0, 160), random.randint(0, 160), random.randint(0, 160))
        if char:
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(debug_frame, f"{char}", (x2+3, y2-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
            cv2.putText(debug_frame, "N", (x2+3, y2-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)

    return results, debug_frame