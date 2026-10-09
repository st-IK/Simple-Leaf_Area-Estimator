import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox
from pathlib import Path

# ============================================================
# 設定
# ============================================================

v_level = 15  # 明るさ
s_level = 100   # 彩度
l_level = 40   # Lab

# ============================================================
# 日本語・Unicodeパス対応
# ============================================================

def imread_unicode(path):
    """Unicode（日本語など）を含むパスから画像を読み込む．"""
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"画像を読み込めませんでした:\n{path}")
    return image


def imwrite_unicode(path, image):
    """Unicode（日本語など）を含むパスへ画像を書き込む．"""
    path = str(path)
    ext = Path(path).suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        ext = ".png"

    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"画像をエンコードできませんでした:\n{path}")

    encoded.tofile(path)


# ============================================================
# 幾何学処理
# ============================================================

def order_points(pts):
    """
    4点を
    左上 → 右上 → 右下 → 左下
    の順に並べる．
    """
    pts = np.asarray(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    rect[1] = pts[np.argmin(d)]  # TR
    rect[3] = pts[np.argmax(d)]  # BL

    return rect


def line_intersection(line1, line2):
    """
    2直線の交点を返す．
    line = (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    a = np.array([
        [x2 - x1, -(x4 - x3)],
        [y2 - y1, -(y4 - y3)]
    ], dtype=np.float64)

    b = np.array([
        x3 - x1,
        y3 - y1
    ], dtype=np.float64)

    det = np.linalg.det(a)
    if abs(det) < 1e-10:
        return None

    t, _ = np.linalg.solve(a, b)

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return np.array([x, y], dtype=np.float32)


def four_point_transform(image, pts, size=1000):
    """
    四隅を正方形へ射影変換．
    """
    rect = order_points(pts)

    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image,
        M,
        (size, size),
        flags=cv2.INTER_CUBIC
    )

    return warped, M


# ============================================================
# 白い四角形の検出
# ============================================================

def make_white_mask(image):
    """
    黒背景上の白い四角を検出するためのマスク．

    RGBではなくHSV + Labを併用し，
    「明るい」「低彩度」の領域を白候補とする．
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)

    # 白：高輝度・低彩度
    hsv_mask = (
        (v >= 150) &
        (s <= 100)
    )

    # LabのLも補助条件として使用
    lab_mask = l >= 150

    mask = (
        hsv_mask &
        lab_mask
    ).astype(np.uint8) * 255

    # 小さなノイズを除去
    kernel_small = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=1
    )

    # 四角の辺に多少の欠損があっても連結させる
    kernel_large = cv2.getStructuringElement(
        cv2.MORPH_RECT, (15, 15)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_large,
        iterations=2
    )

    return mask


def detect_square(image, debug=False):
    """
    黒背景上の白い正方形を検出する．

    基本方針：
      1. 白色候補を抽出
      2. 輪郭だけに依存せず，エッジも併用
      3. 四角形候補を幾何学的に評価
      4. 必要ならminAreaRectを補助的に使用

    戻り値：
      corners, score, debug_image
    """
    h, w = image.shape[:2]
    image_area = h * w

    white_mask = make_white_mask(image)

    # --------------------------------------------------------
    # 方法1：白色領域の輪郭
    # --------------------------------------------------------
    contours, _ = cv2.findContours(
        white_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # あまりに小さいものは除外
        if area < image_area * 0.03:
            continue

        perimeter = cv2.arcLength(cnt, True)

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            cnt,
            0.015 * perimeter,
            True
        )

        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2).astype(np.float32)

            rect = order_points(pts)

            # 辺長
            top = np.linalg.norm(rect[1] - rect[0])
            right = np.linalg.norm(rect[2] - rect[1])
            bottom = np.linalg.norm(rect[2] - rect[3])
            left = np.linalg.norm(rect[3] - rect[0])

            if min(top, right, bottom, left) < 1:
                continue

            width = (top + bottom) / 2
            height = (left + right) / 2

            aspect = width / height

            # 「正方形」に近いほど高得点
            square_score = 1.0 - min(abs(aspect - 1.0), 1.0)

            # 面積が大きいほど有利
            area_score = min(area / image_area, 1.0)

            # 四辺の長さが互いに近い
            sides = np.array([top, right, bottom, left])
            side_ratio = sides.min() / sides.max()

            score = (
                0.50 * square_score +
                0.30 * area_score +
                0.20 * side_ratio
            )

            candidates.append(
                (score, rect, "contour")
            )

    # --------------------------------------------------------
    # 方法2：エッジから最小外接矩形を作る
    # --------------------------------------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 黒背景と白い四角の境界を強く検出
    edges = cv2.Canny(
        gray,
        50,
        150,
        apertureSize=3
    )

    # エッジを少し接続
    edge_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (7, 7)
    )

    edges_closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        edge_kernel,
        iterations=2
    )

    edge_contours, _ = cv2.findContours(
        edges_closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in edge_contours:
        area = cv2.contourArea(cnt)

        if area < image_area * 0.05:
            continue

        rect_data = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect_data)
        box = order_points(box)

        box_area = cv2.contourArea(
            box.astype(np.float32)
        )

        if box_area <= 0:
            continue

        ratio = box_area / image_area

        if ratio < 0.05:
            continue

        sides = np.array([
            np.linalg.norm(box[1] - box[0]),
            np.linalg.norm(box[2] - box[1]),
            np.linalg.norm(box[3] - box[2]),
            np.linalg.norm(box[0] - box[3])
        ])

        aspect = (
            (sides[0] + sides[2]) /
            (sides[1] + sides[3])
        )

        square_score = 1.0 - min(abs(aspect - 1.0), 1.0)
        side_ratio = sides.min() / sides.max()
        area_score = min(ratio, 1.0)

        score = (
            0.45 * square_score +
            0.25 * side_ratio +
            0.30 * area_score
        )

        candidates.append(
            (score, box, "edge")
        )

    if not candidates:
        raise RuntimeError(
            "白い四角形を検出できませんでした．\n"
            "白い四角が画像内に十分大きく写っているか確認してください．"
        )

    # 最も幾何学的に「四角らしい」候補
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    best_score, best_rect, method = candidates[0]

    # --------------------------------------------------------
    # デバッグ画像
    # --------------------------------------------------------
    debug_image = image.copy()

    cv2.polylines(
        debug_image,
        [best_rect.astype(np.int32)],
        True,
        (0, 0, 255),
        4
    )

    for i, p in enumerate(best_rect):
        cv2.circle(
            debug_image,
            tuple(np.round(p).astype(int)),
            10,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            debug_image,
            ["TL", "TR", "BR", "BL"][i],
            tuple(np.round(p).astype(int) + np.array([10, -10])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    if debug:
        print(
            f"Square detection: method={method}, "
            f"score={best_score:.4f}"
        )

    return best_rect, best_score, debug_image


# ============================================================
# 四角内部の対象物検出
# ============================================================

def create_inside_mask(size, margin_ratio=0.015):
    """
    射影変換後の正方形内部だけを解析対象にする．
    外周の補間・縁の影響を除くため少し内側を使う．
    """
    margin = max(1, int(size * margin_ratio))

    mask = np.zeros((size, size), dtype=np.uint8)

    cv2.rectangle(
        mask,
        (margin, margin),
        (size - margin - 1, size - margin - 1),
        255,
        -1
    )

    return mask


def segment_object(warped):
    """
    黒背景 + 白い四角 + 四角内部の物体を想定．

    四角そのものは既に射影変換で正規化されているため，
    内部の「白ではない領域」を対象物候補として扱う．

    色差だけでなく，HSV/Labを併用して判定する．
    """
    size = warped.shape[0]

    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)

    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)

    # 白い背景の推定
    # 白は高L，高V，低Sになりやすい
    # settings
    white_background = (
        (v >= v_level) &    # 明るさ
        (s <= s_level) &    # 彩度
        (l >= l_level)      # Lab
    )

    # 白ではないものを対象物候補とする
    object_mask = (~white_background).astype(
        np.uint8
    ) * 255

    # ただし，外周の補間領域を除外
    inside = create_inside_mask(size)

    object_mask = cv2.bitwise_and(
        object_mask,
        inside
    )

    # ノイズ除去
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9)
    )

    object_mask = cv2.morphologyEx(
        object_mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1
    )

    object_mask = cv2.morphologyEx(
        object_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2
    )

    # 小さな領域を除去
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        object_mask,
        connectivity=8
    )

    cleaned = np.zeros_like(object_mask)

    # 画像サイズに応じたノイズ除去閾値
    min_area = max(20, int(size * size * 0.00005))

    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def compute_area_percentage(binary_mask):
    """
    白い四角の面積に対する対象物面積．
    射影変換後は正方形なので，
    全画素に対する対象画素の割合で求められる．
    """
    inside = binary_mask > 0

    # 正方形内部の面積
    total = np.count_nonzero(
        create_inside_mask(binary_mask.shape[0]) > 0
    )

    obj = np.count_nonzero(
        binary_mask[inside]
    )

    return obj / total * 100.0


# ============================================================
# 解析本体
# ============================================================

def analyze_image(
    path,
    output_dir=None,
    square_size=1000,
    show=True,
    save=True,
    debug=True
):
    path = Path(path)

    if output_dir is None:
        output_dir = path.parent / "analysis_result"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 画像読み込み
    # --------------------------------------------------------
    image = imread_unicode(path)

    # --------------------------------------------------------
    # 白い四角形を検出
    # --------------------------------------------------------
    corners, score, detection_debug = detect_square(
        image,
        debug=debug
    )

    # --------------------------------------------------------
    # 射影変換
    # --------------------------------------------------------
    warped, transform_matrix = four_point_transform(
        image,
        corners,
        size=square_size
    )

    # --------------------------------------------------------
    # 対象物を抽出
    # --------------------------------------------------------
    object_mask = segment_object(warped)

    # --------------------------------------------------------
    # 面積率
    # --------------------------------------------------------
    percentage = compute_area_percentage(
        object_mask
    )

    # --------------------------------------------------------
    # 対象物を重ねた画像
    # --------------------------------------------------------
    overlay = warped.copy()

    overlay[object_mask == 255] = (
        0.55 * overlay[object_mask == 255] +
        0.45 * np.array([0, 0, 255])
    ).astype(np.uint8)

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------
    stem = path.stem

    warped_path = output_dir / f"{stem}_warped.png"
    binary_path = output_dir / f"{stem}_object_mask.png"
    detection_path = output_dir / f"{stem}_square_detection.png"
    overlay_path = output_dir / f"{stem}_overlay.png"

    if save:
        imwrite_unicode(
            warped_path,
            warped
        )

        imwrite_unicode(
            binary_path,
            object_mask
        )

        imwrite_unicode(
            detection_path,
            detection_debug
        )

        imwrite_unicode(
            overlay_path,
            overlay
        )

    # --------------------------------------------------------
    # 表示
    # --------------------------------------------------------
    if show:
        plt.figure(
            figsize=(16, 10)
        )

        plt.suptitle(
            f"Object area: {percentage:.3f}%\n"
            f"Square detection score: {score:.3f}",
            fontsize=16
        )

        plt.subplot(2, 2, 1)
        plt.imshow(
            cv2.cvtColor(
                detection_debug,
                cv2.COLOR_BGR2RGB
            )
        )
        plt.title("Detected white square")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(
            cv2.cvtColor(
                warped,
                cv2.COLOR_BGR2RGB
            )
        )
        plt.title("Perspective-corrected square")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.imshow(
            object_mask,
            cmap="gray",
            vmin=0,
            vmax=255
        )
        plt.title("Detected object")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(
            cv2.cvtColor(
                overlay,
                cv2.COLOR_BGR2RGB
            )
        )
        plt.title(
            f"Object area = {percentage:.3f}%"
        )
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return {
        "percentage": percentage,
        "corners": corners,
        "detection_score": score,
        "warped_image": warped,
        "object_mask": object_mask,
        "overlay": overlay,
        "warped_path": str(warped_path),
        "binary_path": str(binary_path),
        "detection_path": str(detection_path),
        "overlay_path": str(overlay_path)
    }


# ============================================================
# Tkinter
# ============================================================

def main():
    root = Tk()
    root.withdraw()

    try:
        path = filedialog.askopenfilename(
            parent=root,
            title="解析する画像を選択してください",
            filetypes=[
                (
                    "画像ファイル",
                    "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"
                ),
                ("すべてのファイル", "*.*")
            ]
        )

        if not path:
            print("画像が選択されませんでした．")
            return

        print("=" * 60)
        print("画像解析を開始します")
        print(f"入力: {path}")
        print("=" * 60)

        result = analyze_image(
            path,
            square_size=1000,
            show=True,
            save=True,
            debug=True
        )

        print()
        print("=" * 60)
        print(
            f"対象物面積率 : "
            f"{result['percentage']:.4f} %"
        )
        print(
            f"四角形検出スコア : "
            f"{result['detection_score']:.4f}"
        )
        print()
        print("保存先:")
        print(result["warped_path"])
        print(result["binary_path"])
        print(result["detection_path"])
        print(result["overlay_path"])
        print("=" * 60)

        messagebox.showinfo(
            "解析完了",
            f"対象物面積率\n\n"
            f"{result['percentage']:.4f} %\n\n"
            f"結果画像を入力画像と同じ場所の\n"
            f"「analysis_result」フォルダに保存しました．",
            parent=root
        )

    except Exception as e:
        print("エラー:")
        print(e)

        messagebox.showerror(
            "解析エラー",
            str(e),
            parent=root
        )

    finally:
        root.destroy()


if __name__ == "__main__":
    main()
