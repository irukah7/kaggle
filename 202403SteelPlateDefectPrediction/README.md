# Steel Plate Defect Prediction

* 目標
  * 銅板の様々な欠陥の発生確率を予測する
* 評価方法
  * 予測された確率とGroundTruthのターゲットを使用し、ROC曲線下面積を用いて評価される

最終的なスコアを計算するために、AUCは7つの欠陥カテゴリごとに計算され、平均化される。言い換えれば、スコアは各予測列の個々のAUCの平均である。

* 期限
  * 開始日：2024年3月
  * 最終提出日：2024年3月31日

* train.csv
  * Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults
  * 7つのバイナリターゲットデータ
* test.csv
  * 7つのデータそれぞれの確率を予測すること

## 0310 memo
一旦コミット

## 0312
Notebookを読んでコンペの理解を深める：
https://www.kaggle.com/code/lucamassaron/steel-plate-eda-xgboost-is-all-you-need

## 0313
↑引き続きNotebook読み込み＋Notebook作成
