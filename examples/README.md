# RFscorer Examples

[日本語](#rfscorer-サンプル)

Tutorial notebooks for `rfscorer`, available in English and Japanese.
Follow them in order — **Beginner → Practical → Advanced**.

## Beginner

- [tutorial_beginner_en.ipynb](tutorial_beginner_en.ipynb) — A complete end-to-end walkthrough on a small dataset: load a behavior history, `fit()` the empirical product-choice probabilities, `optimize()` under RF constraints, visualize the surfaces, and `transform()` to score recommendations. The quickest way to grasp the `fit → optimize → transform` flow.

## Practical

- [tutorial_practical_en.ipynb](tutorial_practical_en.ipynb) — A realistic workflow that mirrors production use. Splits data chronologically into train/test, builds all 9 models (empirical + optimized), compares their recommendation accuracy with `evaluate()`, and saves/loads the trained model for reuse.

## Advanced

- [tutorial_advanced_fit_rolling_en.ipynb](tutorial_advanced_fit_rolling_en.ipynb) — Tackles the noise you get when empirical probabilities are estimated from a single reference date. Uses `fit_rolling()` to pool counts across multiple reference dates (rolling aggregation), stabilizing the empirical product-choice probabilities, and compares accuracy across all 9 models.
- [tutorial_advanced_int_time_col_en.ipynb](tutorial_advanced_int_time_col_en.ipynb) — For time-series data that has no calendar dates. Shows how to use an integer `time_col` such as session / period / week numbers: map the timeline to consecutive integers and run the same `split_by_date → fit → transform` flow unchanged.
- [tutorial_advanced_day_freq_en.ipynb](tutorial_advanced_day_freq_en.ipynb) — A way to cap the upper bound of frequency. Counts the number of distinct **days** an item was viewed (instead of every view), so frequency is bounded by the observation window, then compares which model performs best under this definition.

> Note: Running the notebooks generates artifacts (CSV / PNG / pkl / zip) in this directory. These are git-ignored and are not part of the repository.

-----

# RFscorer サンプル

[English](#rfscorer-examples)

`rfscorer` のチュートリアル（日英）です。
**初級 → 実践 → 応用** の順に進めることを想定しています。

## 初級編

- [tutorial_beginner_ja.ipynb](tutorial_beginner_ja.ipynb) — 小さなデータセットで一連の流れを通して学びます。行動履歴の読み込み、`fit()` による経験的商品選択確率の推定、RF制約下での `optimize()`、確率曲面の可視化、`transform()` による推薦スコア算出まで。`fit → optimize → transform` という流れを最短で把握できます。

## 実践編

- [tutorial_practical_ja.ipynb](tutorial_practical_ja.ipynb) — 運用を想定した実践的なワークフローです。データを時系列で訓練・テストに分割し、全9種のモデル（経験的＋最適化）を構築、`evaluate()` で推薦精度を比較し、学習済みモデルを保存・ロードして再利用します。

## 応用編

- [tutorial_advanced_fit_rolling_ja.ipynb](tutorial_advanced_fit_rolling_ja.ipynb) — 単一の基準日では経験的確率にノイズが乗りやすい、という課題に対処します。`fit_rolling()` で複数の基準日にわたって件数を積み増す（ローリング集計）ことで経験的商品選択確率を安定させ、全9種モデルの精度も比較します。
- [tutorial_advanced_int_time_col_ja.ipynb](tutorial_advanced_int_time_col_ja.ipynb) — カレンダー概念を持たない時系列データへの対応を解説します。日付文字列ではなく、セッション番号・期番号・週番号などの整数値を `time_col` として扱う方法を示します。時系列を連番整数にマッピングすれば、`split_by_date → fit → transform` の流れはそのまま使えます。
- [tutorial_advanced_day_freq_ja.ipynb](tutorial_advanced_day_freq_ja.ipynb) — 頻度の上限を抑えるための手法です。閲覧の総回数ではなく、商品を閲覧した**日数**を数えることで、頻度を観測期間内に収めます。この定義のもとでどのモデルが最も高精度かも比較します。
