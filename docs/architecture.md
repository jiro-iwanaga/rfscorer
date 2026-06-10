# 技術仕様書

## モジュール構成

| モジュール | クラス | 役割 |
|-----------|--------|------|
| `scorer.py` | `RecencyFrequencyScorer` | 公開 API。fit・transform・optimize・evaluate・plot・export の各メソッドを提供する |
| `optimizer.py` | `RFOptimizer` | 内部モジュール。`optimize()` から委譲された凸2次計画問題を cvxpy で求解する |
| `__init__.py` | — | `RecencyFrequencyScorer` のみを公開する |

## 遅延インポート方針

重い依存ライブラリは使用するメソッドの内部でのみインポートする。`import rfscorer` 自体は軽量に保ち、不要な依存関係をユーザーに強制しない。

| ライブラリ | インポートのタイミング |
|-----------|----------------------|
| `numpy` | `plot_probability_surface()` 呼び出し時 |
| `matplotlib` | `plot_probability_surface()` / `plot_marginal_probability()` 呼び出し時 |
| `cvxpy`（`optimizer.py` 経由） | `optimize()` 呼び出し時 |

この方針により、経験的確率の推定（`fit`・`transform`・`evaluate`）のみを使用するユーザーは `cvxpy` をインストールしなくても動作する。

## 外部依存ライブラリ

| ライブラリ | 用途 |
|-----------|------|
| `pandas` | データ操作・集計・出力形式 |
| `cvxpy` | 凸2次計画問題のソルバー |
| `numpy` | `plot_probability_surface()` 内のグリッド生成（遅延インポート） |
| `matplotlib` | `plot_probability_surface()` / `plot_marginal_probability()` 内の描画・`Figure` 返却（遅延インポート） |

オプション依存 (`pip install rfscorer[ja]`):

| ライブラリ | 用途 |
|-----------|------|
| `japanize-matplotlib` | 軸ラベル・タイトルへの日本語使用（`plot_probability_surface()` / `plot_marginal_probability()`） |

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
