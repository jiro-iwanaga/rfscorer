# リポジトリ構造定義書

## ディレクトリ構成

```
rfscorer/
├── .devcontainer/
│   └── devcontainer.json             # 開発コンテナ設定
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI 設定
├── .gitignore
├── .steering/                        # 作業単位のドキュメント（開発ごとに作成）
│   └── YYYYMMDD-development-title/
│       ├── requirements.md
│       ├── design.md
│       └── tasklist.md
├── CHANGELOG.md                      # 変更履歴
├── LICENSE
├── README.md
├── pyproject.toml                    # パッケージ設定・依存関係
├── uv.lock                           # 依存関係ロック
├── src/
│   └── rfscorer/
│       ├── __init__.py               # 公開 API のエクスポート
│       ├── scorer.py                 # RecencyFrequencyScorer
│       ├── optimizer.py              # RecencyFrequencyOptimizer
│       ├── utils.py                  # 公開ユーティリティ（split_by_date）
│       ├── _plotting.py              # 可視化 Mixin（内部用）
│       └── _time_utils.py            # 時間軸の正規化（内部用）
├── tests/
│   ├── test_scorer.py
│   ├── test_optimizer.py
│   ├── test_utils.py
│   └── test_save_load.py
├── examples/
│   ├── tutorial_beginner_en.ipynb        # 初級チュートリアル（英語）
│   └── tutorial_beginner_ja.ipynb        # 初級チュートリアル（日本語）
├── img/                              # README 向け画像
└── docs/                             # 永続的ドキュメント
    ├── product-requirements.md       # プロダクト要求定義書
    ├── functional-design.md          # 機能設計書
    ├── architecture.md               # 技術仕様書
    ├── repository-structure.md       # リポジトリ構造定義書（本書）
    ├── development-guidelines.md     # 開発ガイドライン
    └── glossary.md                   # ユビキタス言語定義
```

## 各ファイル・ディレクトリの役割

### ルート

| パス | 説明 |
|------|------|
| `pyproject.toml` | パッケージのメタデータ、依存関係、ビルド設定、ruff 設定を管理する |
| `uv.lock` | `uv` が生成する依存関係ロックファイル。バージョン管理に含める |
| `README.md` | PyPI および GitHub 向けのパッケージ説明。インストール方法と使用例を含む |
| `CHANGELOG.md` | バージョンごとの変更履歴 |

### `.devcontainer/`

| パス | 説明 |
|------|------|
| `devcontainer.json` | VS Code Dev Containers / GitHub Codespaces 向けの開発コンテナ設定 |

### `.github/`

| パス | 説明 |
|------|------|
| `workflows/ci.yml` | GitHub Actions による CI 設定（テスト・リント） |

### `src/rfscorer/`

| パス | 説明 |
|------|------|
| `__init__.py` | `RecencyFrequencyScorer` と `split_by_date` を公開 API としてエクスポートする |
| `scorer.py` | `RecencyFrequencyScorer` クラスを実装する |
| `optimizer.py` | `RecencyFrequencyOptimizer` クラスを実装する。RF 単調性制約付き凸2次計画問題のモデル構築・求解を担う |
| `utils.py` | `split_by_date()` など、データ準備用の公開ユーティリティを提供する |
| `_plotting.py` | `PlottingMixin` を実装する。`RecencyFrequencyScorer` に Mixin として継承され、可視化メソッドを提供する（内部用） |
| `_time_utils.py` | 時間軸の正規化・変換など、内部用のヘルパー関数（プライベート） |

### `tests/`

| パス | 説明 |
|------|------|
| `test_scorer.py` | `RecencyFrequencyScorer` のユニットテスト |
| `test_optimizer.py` | `RecencyFrequencyOptimizer` のユニットテスト |
| `test_utils.py` | `split_by_date()` など、ユーティリティのユニットテスト |
| `test_save_load.py` | `save()` / `load()` / `save_zip()` / `load_zip()` のユニットテスト |

### `examples/`

| パス | 説明 |
|------|------|
| `tutorial_beginner_en.ipynb` | 初級チュートリアル（英語）。fit・optimize・transform・evaluate・save・load の基本ワークフローを解説 |
| `tutorial_beginner_ja.ipynb` | 初級チュートリアル（日本語）。同上 |


### `img/`

README.md に埋め込む確率面ワイヤーフレームの画像を配置する。

### `docs/`

永続的なプロジェクトドキュメント。基本方針や設計が変わらない限り頻繁には更新しない。

| パス | 説明 |
|------|------|
| `product-requirements.md` | プロダクトの目的・ターゲットユーザー・機能要求・制約 |
| `functional-design.md` | アルゴリズムの定義・クラス仕様・入出力仕様 |
| `architecture.md` | パッケージ構成・外部依存・ビルド・配布方法 |
| `repository-structure.md` | 本書 |
| `development-guidelines.md` | コーディング規約・テスト方針・リリース手順 |
| `glossary.md` | ユビキタス言語（ドメイン用語の定義） |

### `.steering/`

個別の開発作業ごとのドキュメント。ディレクトリ名は `YYYYMMDD-development-title` 形式。  
作業完了後も意思決定の経緯として保持する。

| ファイル | 説明 |
|---------|------|
| `requirements.md` | 作業で実現すること・制約・受け入れ条件 |
| `design.md` | 実装方針・変更対象・影響範囲 |
| `tasklist.md` | 実装タスク・進捗・完了条件 |
