# design.md

## 実装アプローチ

2 種類の保存形式を提供する。

- **pickle形式** (`save` / `load`): 高速・軽量。サーバーデプロイ向け。
- **zip形式** (`save_zip` / `load_zip`): CSV・PNG を同梱。研究者間の共有・成果物管理向け。

```python
# pickle形式
scorer.save()                        # → ./rfscorer.pkl
scorer.save("models/")               # → models/rfscorer.pkl
scorer.save("models/my_scorer.pkl")
scorer = RecencyFrequencyScorer.load("models/my_scorer.pkl")

# zip形式
scorer.save_zip()                    # → ./rfscorer.zip
scorer.save_zip("artifacts/")        # → artifacts/rfscorer.zip
scorer.save_zip("artifacts/v1.zip")
scorer = RecencyFrequencyScorer.load_zip("artifacts/v1.zip")
```

## pickle形式の保存フォーマット

pickle でシリアライズするオブジェクトはラッパー辞書とし、バージョン情報を付与する。

```python
{
    "rfscorer_version": "0.4.3",
    "scorer": self,
}
```

## zip形式のアーカイブ構成

```
rfscorer.zip
├── metadata.json           # バージョン・パラメータ・統計情報
├── rfscorer.pkl            # モデル本体（load_zip() はここから復元）
├── probabilities/
│   ├── emp_probability.csv     # fit() 後に常に含む
│   ├── er_probability.csv
│   ├── ef_probability.csv
│   ├── mono_probability.csv    # optimize(kind="mono") 済みなら含む
│   ├── mr_probability.csv      # 以下同様、呼んだ種類のみ含む
│   ├── mf_probability.csv
│   ├── mrc_probability.csv
│   ├── mfc_probability.csv
│   └── mcc_probability.csv
└── plots/
    ├── emp_surface.png         # fit() 後に常に含む（2D surface）
    ├── er_marginal.png         # recency 限界確率（折れ線）
    ├── ef_marginal.png         # frequency 限界確率（折れ線）
    ├── mono_surface.png        # optimize 済みなら含む（以下同様）
    ├── mr_marginal.png
    ├── mf_marginal.png
    ├── mrc_surface.png
    ├── mfc_surface.png
    └── mcc_surface.png
```

`metadata.json` の内容：

```json
{
  "rfscorer_version": "0.4.3",
  "user_col": "user",
  "item_col": "item",
  "time_col": "datetime",
  "unit": 1,
  "recency_limit": 7,
  "frequency_limit": 3,
  "observation_start": "2024-01-01",
  "observation_end": "2024-01-07",
  "record_num": 7,
  "total_cv": 2,
  "optimized_kinds": ["mono", "mr"]
}
```

CSV は既存の `export_probability_csv()` の出力形式と同一。
PNG は既存の `plot_probability_surface()` / `plot_marginal_probability()` を内部で呼び出して生成。

## バージョン互換性

両形式共通。バージョンは `importlib.metadata.version("rfscorer")` から取得する。
パッチ違い（`0.4.0` → `0.4.3`）は無視し、マイナー違い（`0.4.x` → `0.5.x`）・メジャー違い（`0.x` → `1.x`）は `UserWarning` を出してロードを続行する。

## 変更するコンポーネント

| 対象 | 内容 |
|---|---|
| `src/rfscorer/scorer.py` | `save()` / `load()` / `save_zip()` / `load_zip()` の 4 メソッド追加 |
| `__init__.py` | 変更なし |
| 既存のパブリック API | 変更なし（後方互換性を維持） |
| 依存関係 | 変更なし（`pickle`, `zipfile`, `json`, `pathlib`, `importlib.metadata` はすべて標準ライブラリ） |
| テスト | `tests/test_save_load.py` に zip 関連テストを追記 |

## テスト設計

### pickle形式（実装済み）

| テストケース | 確認内容 |
|---|---|
| `test_save_load_after_fit` | fit 後に保存→ロードし、`predict()` の結果が一致する |
| `test_save_load_after_optimize` | optimize 後に保存→ロードし、最適化済みの `predict()` の結果が一致する |
| `test_save_path_none` | `path=None` でカレントディレクトリに `rfscorer.pkl` が生成される |
| `test_save_path_directory` | ディレクトリを指定するとそのディレクトリ内に `rfscorer.pkl` が生成される |
| `test_save_path_file` | ファイルパスを指定すると指定名で保存される |
| `test_load_version_mismatch` | マイナー違いのバージョンで保存したファイルのロードで `UserWarning` が出る |
| `test_path_accepts_string` | `str` で動作する |
| `test_path_accepts_pathlib` | `pathlib.Path` で動作する |

### zip形式（追加予定）

| テストケース | 確認内容 |
|---|---|
| `test_save_zip_load_zip_after_fit` | fit 後に保存→ロードし、`predict()` の結果が一致する |
| `test_save_zip_load_zip_after_optimize` | optimize 後に保存→ロードし、最適化済みの `predict()` の結果が一致する |
| `test_save_zip_path_none` | `path=None` でカレントディレクトリに `rfscorer.zip` が生成される |
| `test_save_zip_path_directory` | ディレクトリを指定するとそのディレクトリ内に `rfscorer.zip` が生成される |
| `test_save_zip_path_file` | ファイルパスを指定すると指定名で保存される |
| `test_save_zip_contents_after_fit` | zip 内に metadata.json・pkl・CSV 3本・PNG 3本が存在する |
| `test_save_zip_contents_after_optimize` | optimize 済みの種類に対応する CSV・PNG が追加される |
| `test_save_zip_version_mismatch` | マイナー違いのバージョンで `UserWarning` が出る |
