# CLAUDE.md

## 概要

本ファイルはGitHubを利用したPythonモジュール`rfscorer` の開発をするための指示書である。 
`rfscorer` は、Recency-Frequency に基づく商品推薦スコアリングを提供する Python パッケージであり、PyPI で公開して、pip installが可能にすることを前提に開発する。

## プロジェクトルール

* 本プロジェクトは Python パッケージとして開発する。
* Python コマンドは原則として `uv` を利用する。
* 明示的に依頼されない限り、`pip install` を利用しない。
* コードは `src/rfscorer/` に配置する。
* テストコードは `tests/` に配置する。
* 変更後は、必要に応じてテスト・リント・フォーマットを実行する。

## プロジェクト構成

* ライブラリコード: `src/rfscorer/`
* テスト: `tests/`
* サンプル: `examples/`
* 永続的ドキュメント: `docs/`
* 作業単位のドキュメント: `.steering/`
* ビルド設定: `pyproject.toml`
* 依存関係ロック: `uv.lock`

## コマンド

* 依存関係のインストール: `uv sync --dev`
* テスト実行: `uv run pytest`
* リント実行: `uv run ruff check .`
* フォーマット実行: `uv run ruff format .`
* パッケージビルド: `uv build`

## 開発メモ

* このパッケージは Recency-Frequency based recommendation scoring を提供する。
* RF は Random Forest ではなく Recency-Frequency を意味する。
* 主な用途は商品推薦とリピート購買モデリングである。
* 推薦システム全体ではなく、推薦スコアや購買確率を計算する部品として設計する。
* 公開クラスは `rfscorer` から import できるようにする。
* 公開 API はシンプルに保ち、README または docstring で説明する。
* 解釈可能性を重視し、ブラックボックス化しない。

## 想定する公開クラス

* `RecencyFrequencyScorer`

## ドキュメント管理

詳細は [`docs/development-guidelines.md`](docs/development-guidelines.md) を参照。

* `docs/`: 永続的ドキュメント（要求定義・機能設計・技術仕様など）
* `.steering/YYYYMMDD-development-title/`: 作業単位のドキュメント（requirements / design / tasklist）