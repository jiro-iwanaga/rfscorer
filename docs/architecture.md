# 技術仕様書

## モジュール構成

| モジュール | クラス・関数 | 役割 |
|-----------|--------|------|
| `scorer.py` | `RecencyFrequencyScorer` | 公開 API。fit・predict・transform・optimize・evaluate・plot・export の各メソッドを提供する |
| `optimizer.py` | `RecencyFrequencyOptimizer` | 内部モジュール。`optimize()` から委譲された凸2次計画問題を cvxpy で求解する |
| `utils.py` | `split_by_date()` | 公開ユーティリティ。観測ログと評価ログを target_date で自動分割する |
| `_time_utils.py` | `normalize_ref()`, `normalize_sequence_col()` | 内部用。時間軸の正規化・変換（プライベート） |
| `__init__.py` | — | `RecencyFrequencyScorer` と `split_by_date` を公開 API としてエクスポートする |

## 遅延インポート方針

重い依存ライブラリは使用するメソッドの内部でのみインポートする。`import rfscorer` 自体は軽量に保ち、不要な依存関係をユーザーに強制しない。

| ライブラリ | インポートのタイミング |
|-----------|----------------------|
| `matplotlib` | `plot_probability_surface()` / `plot_marginal_probability()` 呼び出し時 |
| `cvxpy`（`optimizer.py` 経由） | `optimize()` 呼び出し時 |

この方針により、経験的確率の推定（`fit`・`transform`・`evaluate`）のみを使用するユーザーは `cvxpy` をインストールしなくても動作する。

## 外部依存ライブラリ

| ライブラリ | 用途 |
|-----------|------|
| `pandas` | データ操作・集計・出力形式 |
| `cvxpy` | 凸2次計画問題のソルバー |
| `numpy` | 型判定（`_normalize_ref()`）・`plot_probability_surface()` 内のグリッド生成 |
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
