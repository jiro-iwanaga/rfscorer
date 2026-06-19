# RFscorer Examples

[日本語](#rfscorer-サンプル)

Runnable tutorial notebooks for `rfscorer`, available in English and Japanese.
Follow them in order — **Beginner → Practical → Advanced**.

## Beginner

- [tutorial_beginner_en.ipynb](tutorial_beginner_en.ipynb) — end-to-end walkthrough: load data, fit, optimize, visualize, transform, and evaluate

## Practical

- [tutorial_practical_en.ipynb](tutorial_practical_en.ipynb) — practical workflow: chronological train/test split, build all 9 models, compare accuracy, and save/load the model

## Advanced

- [tutorial_advanced_fit_rolling_en.ipynb](tutorial_advanced_fit_rolling_en.ipynb) — time-series rolling training with `fit_rolling()` to stabilize empirical probabilities across multiple reference dates
- [tutorial_advanced_int_time_col_en.ipynb](tutorial_advanced_int_time_col_en.ipynb) — handle an integer `time_col` (session / period numbers) instead of date strings, for time-series data without a calendar
- [tutorial_advanced_day_freq_en.ipynb](tutorial_advanced_day_freq_en.ipynb) — aggregate frequency by number of days viewed (instead of number of views) to cap its upper bound, then compare model accuracy

> Note: Running the notebooks generates artifacts (CSV / PNG / pkl / zip) in this directory. These are git-ignored and are not part of the repository.

-----

# RFscorer サンプル

[English](#rfscorer-examples)

`rfscorer` の実行可能なチュートリアル（日英）です。
**初級 → 実践 → 応用** の順に進めることを想定しています。

## 初級編

- [tutorial_beginner_ja.ipynb](tutorial_beginner_ja.ipynb) — 初級編：データロード、モデル構築・最適化・可視化、推薦スコア算出、精度評価までのコードを紹介します。

## 実践編

- [tutorial_practical_ja.ipynb](tutorial_practical_ja.ipynb) — 実践編：時系列での訓練・テスト分割、全9種のモデル構築と精度比較、モデルの保存・ロードを紹介します。

## 応用編

- [tutorial_advanced_fit_rolling_ja.ipynb](tutorial_advanced_fit_rolling_ja.ipynb) — 応用編：`fit_rolling()` で複数の基準日にわたるローリング集計を行うことで経験的商品選択確率を安定させます。全9種モデルの精度比較も含みます。
- [tutorial_advanced_int_time_col_ja.ipynb](tutorial_advanced_int_time_col_ja.ipynb) — 応用編：日付文字列ではなく整数値の `time_col`（セッション番号・期番号など）に対応する方法を紹介します。カレンダー概念を持たない時系列データ向けです。
- [tutorial_advanced_day_freq_ja.ipynb](tutorial_advanced_day_freq_ja.ipynb) — 応用編：頻度を「閲覧回数」ではなく「閲覧日数」で集計して頻度の上限を抑える方法と、モデル間の精度比較を紹介します。

> 注：ノートブックを実行すると、このディレクトリに生成物（CSV / PNG / pkl / zip）が作られます。これらは git 管理対象外です。
