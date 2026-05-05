import cv2
import pytesseract
import numpy as np
import concurrent.futures
from queue import Queue
import os
import re

# NOTE: Set your Tesseract path before initializing the pool if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
DEBUG = True

class TesseractThreadPool:
    """
    Manages a pool of individual Tesseract engine instances to allow 
    safe multithreaded character recognition on multiple ROIs.
    """
    def __init__(self, pool_size=4):
        # PSM 13: Raw line mode. Better for 1-2 digits than PSM 10 (Single Char).
        # PSM 7 (Single line) is also a strong alternative if 13 struggles.
        self.config = r'--psm 13 -c tessedit_char_whitelist=0123456789X<()>'
        self.engine_pool = Queue()
        self.pool_size = pool_size

        # Initialize individual engine handles. These handle the connection 
        # to the Tesseract binary separately to avoid crashes.
        for _ in range(pool_size):
            # We must pass the config during initialization for some Tesseract APIs
            self.engine_pool.put(self.config)

    def process_roi(self, roi):
        """Worker thread function: borrows an engine and OCRs the ROI."""
        # 1. Borrow an engine config connection
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
                    # Clean text to strictly match whitelist (prevents ghost characters)
                    clean_text = re.sub(r'[^0123456789X<()>]', '', text)
                    if clean_text:
                        text_segments.append(clean_text)
                        confidences.append(int(data['conf'][j]))
            
            if text_segments:
                best_char = "".join(text_segments)
                best_conf = int(np.mean(confidences)) # Average confidence for multi-digit
            
            # Debugging for failed reads
            if best_char == "" and DEBUG:
                if not os.path.exists("debug_ocr"): os.makedirs("debug_ocr")
                cv2.imwrite(f"debug_ocr/fail_{np.random.randint(1000)}.png", roi)
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
                # 1. Binarization (Crucial for Tesseract)
                # Ensure black text on white background
                edge_mask = np.ones(roi.shape, dtype=bool)
                edge_mask[1:-1, 1:-1] = False
                if np.mean(roi[edge_mask]) < 127: 
                    roi = cv2.bitwise_not(roi)
                
                # 2. Resize to a height of about 30-50 pixels (Tesseract's sweet spot)
                scaling = 45.0 / roi.shape[0]
                roi = cv2.resize(roi, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_CUBIC)
                
                # 3. Clean threshold
                _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 4. Standardized Padding (The 'Moat')
                roi = cv2.copyMakeBorder(roi, padding_px, padding_px, padding_px, padding_px, 
                                        cv2.BORDER_CONSTANT, value=[255, 255, 255])
                
                pre_processed_rois.append(roi)
                roi_metadata.append({
                    'string_idx': i,
                    'x': x_pos,
                    'bbox': (int(x), int(y), int(x + w), int(y + h))
                }) 

    if not pre_processed_rois: return results, debug_frame
    results = [[] for _ in range(6)]

    # --- 2. Concurrent Character Recognition ---
    # Submit all pre-processed ROIs to the thread pool for simultaneous processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=_ocr_pool.pool_size) as executor:
        # map ensures that ocr_results are returned in the exact same order as pre_processed_rois
        ocr_results = list(executor.map(_ocr_pool.process_roi, pre_processed_rois))

    # --- 3. Assemble Results and Draw Debug ---
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

        if char:
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(debug_frame, f"{char}", (x1+w+7, y1+h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
            cv2.putText(debug_frame, "N", (x1+w+7, y1+h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)

    return results, debug_frame