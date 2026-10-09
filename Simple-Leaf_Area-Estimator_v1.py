import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # top-left
    rect[2] = pts[np.argmax(s)]    # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect

def four_point_transform(image, pts, size=600):
    rect = order_points_clockwise(pts)
    (tl, tr, br, bl) = rect
    maxSide = size
    dst = np.array([
        [0, 0],
        [maxSide - 1, 0],
        [maxSide - 1, maxSide - 1],
        [0, maxSide - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxSide, maxSide))
    return warped, M

def detect_white_square_contour(image, debug=False):
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 200])   
    upper = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:6]:
        area = cv2.contourArea(cnt)
        if area < 0.01 * h * w:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4,2).astype(np.float32)
            rect = order_points_clockwise(pts)
            (tl, tr, br, bl) = rect
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            wmean = (widthA + widthB) / 2.0
            hmean = (heightA + heightB) / 2.0
            if wmean == 0 or hmean == 0:
                continue
            ar = wmean / hmean
            if 0.7 <= ar <= 1.3:
                return pts
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)

def binarize_object(warped_color, debug=False):
    gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mean_fg = np.mean(gray[th==255]) if np.any(th==255) else 255
    mean_bg = np.mean(gray[th==0]) if np.any(th==0) else 0
    if mean_fg > mean_bg:
        th = cv2.bitwise_not(th)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    return th

def compute_area_percentage(binary_mask):
    total = binary_mask.size
    obj = np.count_nonzero(binary_mask == 255)
    return (obj / total) * 100.0

def analyze_image(path, output_prefix="out", square_size=800, debug=False, show=True):
    img = cv2.imdecode(
    np.fromfile(path, dtype=np.uint8),
    cv2.IMREAD_COLOR
    )
    if img is None:
        raise FileNotFoundError(f"Cannot read image {path}")
    orig = img.copy()
    pts = detect_white_square_contour(img, debug=debug)
    if pts is None:
        raise RuntimeError("Could not detect the white square.")
    warped_color, M = four_point_transform(orig, pts, size=square_size)
    binary = binarize_object(warped_color, debug=debug)
    percent = compute_area_percentage(binary)
    warped_bgr_path = f"{output_prefix}_warped.png"
    binary_path = f"{output_prefix}_binary.png"
    cv2.imencode(".png", warped_color)[1].tofile(warped_bgr_path)
    cv2.imencode(".png", binary)[1].tofile(binary_path)
    if show:
        plt.figure(figsize=(10,6))
        plt.suptitle(f"Object area: {percent:.2f}% of square", fontsize=14)
        plt.subplot(1,2,1)
        plt.imshow(cv2.cvtColor(warped_color, cv2.COLOR_BGR2RGB))
        plt.title("Perspective-corrected square")
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(binary, cmap='gray')
        plt.title("Binarized object (white=object)")
        plt.axis('off')
        plt.show()
    return {
        "percentage": percent,
        "warped_path": warped_bgr_path,
        "binary_path": binary_path,
        "warped_image": warped_color,
        "binary_image": binary
    }

if __name__ == "__main__":
    # ファイル選択ダイアログを開く
    Tk().withdraw()  # Tkのメインウィンドウを隠す
    path = filedialog.askopenfilename(
        title="画像を選択してください",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not path:
        print("画像が選択されませんでした")
        exit()

    try:
        res = analyze_image(path, output_prefix="result", square_size=800, debug=True, show=True)
        print(f"Computed area: {res['percentage']:.4f}%")
    except Exception as e:
        print("エラー:", e)

# "C:\Users\J.J Gurton\OneDrive\画像\カメラ ロール\WIN_20261001_16_50_21_Pro.jpg"