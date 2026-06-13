# 開発ガイドライン

## 開発環境

- Python バージョン: 3.10 以上
- パッケージ管理: `uv`（`pip install` は原則使用しない）
- 依存関係のインストール: `uv sync --dev`

## コマンド

| 操作 | コマンド |
|------|---------|
| 依存関係のインストール | `uv sync --dev` |
| テスト実行 | `uv run pytest` |
| 単一テスト実行 | `uv run pytest tests/test_scorer.py::ClassName::test_method_name` |
| リント | `uv run ruff check .` |
| フォーマット | `uv run ruff format .` |
| パッケージビルド | `uv build` |
| PyPI 公開 | `uv publish` |
| Jupyter カーネル登録（初回のみ） | `uv run python -m ipykernel install --user --name rfscorer --display-name "rfscorer (venv)"` |
| JupyterLab 起動 | `uv run jupyter lab --IdentityProvider.token='' --PasswordIdentityProvider.hashed_password=''` |

> **JupyterLab 起動について**
> - 起動時にデフォルトブラウザで `http://localhost:8888/lab` が自動的に開く
> - 起動ログ・URL は stderr に出力される
> - 停止は端末で `Ctrl+C`
> - ヘッドレス環境（リモート SSH・devcontainer など）では `--no-browser` を追記する

## コーディング規約

- フォーマッタ・リンタは `ruff` を使用する（設定は `pyproject.toml` で管理）
- 公開クラス・メソッドには docstring を記述する
- プライベートなモジュール・関数には `_` プレフィックスをつける
- 公開 API はシンプルに保ち、内部実装を隠蔽する

## API 設計方針

- scikit-learn スタイル（`fit` / `transform` / `optimize`）に準拠する。期間指定によるデータ準備は `rfscorer.split_by_date()` ユーティリティで分離する
- 推定後の属性名は末尾に `_` をつける（例: `emp_probability_`）
- `from rfscorer import RecencyFrequencyScorer` でインポートできるようにする
- 解釈可能性を重視し、スコアの算出根拠を説明できる設計にする

## テスト方針

- テストコードは `tests/` に配置する
- `pytest` を使用する
- `fit()`・`predict()`・`transform()`・`evaluate()`・`optimize()`・`plot_probability_surface()`・`plot_marginal_probability()`・`export_probability_csv()` の正常系・異常系をカバーする
- `split_by_date()` ユーティリティの正常系・異常系をカバーする

## ドキュメント管理

### 永続的ドキュメント: `docs/`

基本方針や設計が変わらない限り、頻繁には更新しない。  
基本設計に影響する変更を実施した際は、対応するドキュメントも合わせて更新する。

| ファイル | 内容 |
|---------|------|
| `product-requirements.md` | プロダクト要求定義書 |
| `functional-design.md` | 機能設計書 |
| `architecture.md` | 技術仕様書 |
| `repository-structure.md` | リポジトリ構造定義書 |
| `development-guidelines.md` | 本書 |
| `glossary.md` | ユビキタス言語定義 |

### 作業単位のドキュメント: `.steering/`

重要な判断をともなう開発をする際に作成する。作業完了後も意思決定・変更経緯の記録として保持する。  
ディレクトリ名は `YYYYMMDD-development-title` 形式。

```
.steering/20260602-initial-implementation/
.steering/20260610-add-optimized-scorer/
```

各作業ディレクトリに必要に応じて以下を作成する。

| ファイル | 内容 |
|---------|------|
| `requirements.md` | 実現すること・制約・受け入れ条件 |
| `design.md` | 実装方針・変更対象・影響範囲 |
| `tasklist.md` | 実装タスク・進捗・完了条件 |

## 開発プロセス

1. 変更内容の既存設計（`docs/`）への影響を確認する
2. `.steering/YYYYMMDD-development-title/` を作成する
3. `requirements.md`・`design.md`・`tasklist.md` に作業内容を整理する
4. `tasklist.md` に基づいて実装を進める
5. 基本設計に影響する変更は `docs/` も更新する
6. 実装後にテスト・リント・フォーマットを確認する

## リリース手順

1. `pyproject.toml` のバージョンを更新する
2. `uv build` でパッケージをビルドする
3. `uv publish` で PyPI に公開する
4. GitHub にバージョンタグを付けてプッシュする（例: `git tag v0.1.0 && git push origin v0.1.0`）
