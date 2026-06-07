# 技術仕様書

## 外部依存ライブラリ

| ライブラリ | 用途 |
|-----------|------|
| `pandas` | データ操作・集計・出力形式 |
| `cvxpy` | 凸2次計画問題のソルバー |
| `numpy` | `plot_probability_surface()` 内のグリッド生成（遅延インポート） |
| `matplotlib` | `plot_probability_surface()` 内の3次元ワイヤーフレーム描画・`Figure` 返却（遅延インポート） |

開発依存:

| ライブラリ | 用途 |
|-----------|------|
| `pytest` | テスト |
| `ruff` | リント・フォーマット |
| `jupyterlab` | `examples/` の Jupyter ノートブック実行環境 |
| `ipykernel` | JupyterLab 用 Python カーネル |

## ビルド・配布

- パッケージ管理: `uv`、設定は `pyproject.toml`
- Python バージョン: 3.10 以上を想定
