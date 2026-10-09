# Simple Leaf Area Estimator

白い正方形の基準領域を写した画像から葉を抽出し，正方形の面積に対する葉の面積率を簡易的に計算するPythonツールです．画像内の正方形を検出して射影変換で補正するため，斜めから撮影した画像も解析できます．

## 特徴

- 画像内の白い正方形を検出し，正面から見た正方形に補正
- 正方形内部の葉などの対象物を抽出し，面積率（%）を計算
- 検出結果，補正画像，抽出マスク，重ね合わせ画像を保存
- 日本語を含むファイルパスに対応
- 画像選択ダイアログから操作可能

## 動作の前提

このツールは，次の条件を想定した簡易解析です．

- 葉を，白い正方形の内側に置いて撮影する
- 白い正方形が画像内で十分な大きさで，四隅が見えるようにする
- 正方形と葉の色の違いを画像から識別できるようにする
- 影や反射，背景の白い物体など，検出を妨げる要素をできるだけ避ける

得られる値は**正方形の面積に対する葉の面積率（%）**です．実際の面積（cm²など）を測定する機能ではありません．既知の実寸の正方形を基準にすれば，面積率から葉の面積を換算できます．

## 必要な環境

- Python 3
- NumPy
- OpenCV
- Matplotlib
- Tkinter（通常のPythonインストールに含まれます）

## セットアップ

仮想環境を作成して有効化した後，必要なライブラリをインストールします．

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

ライブラリをインストールします．

```bash
python -m pip install numpy opencv-python matplotlib
```

## 使い方

現在の推奨版である`Simple-Leaf_Area-Estimator_v2.py`を実行します．

```bash
python Simple-Leaf_Area-Estimator_v2.py
```

画像選択ダイアログで解析する画像を選ぶと，解析結果が表示され，入力画像と同じ場所に`analysis_result`フォルダが作成されます．

## 出力ファイル

入力画像のファイル名を`sample.jpg`とした場合，`analysis_result`フォルダに次の画像が保存されます．

| ファイル | 内容 |
| --- | --- |
| `sample_square_detection.png` | 検出した正方形の位置と四隅 |
| `sample_warped.png` | 射影変換で正面から見た形に補正した画像 |
| `sample_object_mask.png` | 葉などの対象物として抽出された領域 |
| `sample_overlay.png` | 抽出領域を元画像に重ねた確認用画像 |

解析時には，対象物面積率と正方形検出スコアも表示されます．検出結果に問題がないか，保存された画像も確認してください．

## Pythonから呼び出す

`analyze_image`関数を利用して，画像パスを指定して解析することもできます．

```python
import importlib.util
from pathlib import Path

script = Path("Simple-Leaf_Area-Estimator_v2.py")
spec = importlib.util.spec_from_file_location("leaf_area_estimator", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

result = module.analyze_image("path/to/image.jpg")
print(f"葉の面積率: {result['percentage']:.4f}%")
```

`analyze_image`には，必要に応じて出力先や表示・保存の設定を指定できます．

```python
result = module.analyze_image(
    "path/to/image.jpg",
    output_dir="results",
    square_size=1000,
    show=False,
    save=True,
)
```

## バージョン

- `Simple-Leaf_Area-Estimator_v2.py`: 正方形検出と対象物抽出を改善した推奨版
- `Simple-Leaf_Area-Estimator_v1.py`: 初期版

## 注意

本ツールは画像処理による簡易推定です．撮影条件，照明，葉の色，背景，正方形の検出精度によって結果が変わります．研究・報告などに使用する場合は，マスク画像を目視確認し，必要に応じて既知面積の試料で精度を検証してください．