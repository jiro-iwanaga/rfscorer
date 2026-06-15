# tasklist.md

## タスク一覧

### pickle形式（完了）
- [x] 1. `src/rfscorer/scorer.py` に `save()` メソッドを追加
- [x] 2. `src/rfscorer/scorer.py` に `load()` クラスメソッドを追加
- [x] 3. `tests/test_save_load.py` を新規作成しテストを実装（8件）
- [x] 4. テスト・リント・フォーマットを実行して品質チェック

### zip形式（追加実装）
- [ ] 5. `src/rfscorer/scorer.py` に `save_zip()` メソッドを追加
- [ ] 6. `src/rfscorer/scorer.py` に `load_zip()` クラスメソッドを追加
- [ ] 7. `tests/test_save_load.py` に zip 関連テストを追記（8件）
- [ ] 8. テスト・リント・フォーマットを実行して品質チェック

## 完了条件

- `scorer.save()` / `scorer.save("dir/")` / `scorer.save("path/to/file.pkl")` がすべて動作する
- `RecencyFrequencyScorer.load("path.pkl")` でインスタンスが復元できる
- `scorer.save_zip()` / `scorer.save_zip("dir/")` / `scorer.save_zip("path/to/file.zip")` がすべて動作する
- `RecencyFrequencyScorer.load_zip("path.zip")` でインスタンスが復元できる
- zip 内に metadata.json・rfscorer.pkl・CSV・PNG が正しく含まれる
- fit 後・optimize 後のいずれも保存・ロード後に `predict()` が同じ結果を返す
- バージョン違い時に `UserWarning` が出る
- `uv run pytest` が全テストパス
- `uv run ruff check .` と `uv run ruff format .` がエラーなし
