# Tabular Playground Series - Jul 2022
## 概要
この課題では、各行が特定のクラスタに属するデータセットが与えられます。目的は各行が属するクラスタを予測することです。訓練データを与えられていない且つグランドトゥルースラベルの中にいくつのクラスタがあるのかも教えられていない。教師なし学習である。
![](2022-07-03-21-29-20.png)

## 評価方法
提出データは真のクラスタラベルと予測したクラスタラベルの間の[Adjusted Rand Index](https://en.wikipedia.org/wiki/Rand_index)で評価される。この問題では、クラスタの数や学習ラベルは与えられない。

# log

# 7/4、7/5
* データの調査
* Notebooks
    * わかりやすそうなEDA
    * https://www.kaggle.com/code/kartushovdanil/tps-jul-22-advanced-2-sol

* 7/6
* Notebooks
    * 丁寧にEDAからクラスタリング・外れ値まで説明されている
    * https://www.kaggle.com/code/javigallego/outliers-eda-clustering-tutorial

# 7/7
* 7月のTPSは教師なし学習でクラスタリングを使用する方針がオーソドックスになる

# 7/12, 7/13
* BayesianGaussianMixtureを使ってモデリングした良記事があったので参考にしてNotebook作成
* https://www.kaggle.com/hikarumoriya/tps-jul2022-eda/edit
* 結果
    * ![](2022-07-14-09-38-13.png)

# 7/15
* Discussion
    * 特徴量を減らすことでスコアが向上した
    * https://www.kaggle.com/competitions/tabular-playground-series-jul-2022/discussion/334875

# 7/16
* sns.heatmap()
    * 数値を表示: 引数`annot`
    * カラーバー表示・非表示: 引数`cbar`
    * 正方形で表示: 引数`square`
    * 最大値、最小値、中央値を指定: 引数`vmax`, `vmin`, `center`
    * 色（カラーマップ）を指定: 引数`cmap`
    * サイズを指定
        * sns.heatmap()の引数ではないが一応説明
        * 生成される画像のサイズはfigsize(インチ)とdpi(インチあたりのドット数)で決定される
        * plt.figure(figsize=(9, 6))

# 07/19, 7/20
* Notebooks
    * [Clusters and LGBの簡単な手法紹介](https://www.kaggle.com/code/ricopue/tps-jul22-clusters-and-lgb)
    * [結構難しいこと書いてるけど気になるので見てみた](https://www.kaggle.com/code/samuelcortinhas/poisson-hybrid-mixture-models)
    * [クラスタリングアルゴリズムの紹介](https://www.kaggle.com/competitions/tabular-playground-series-jul-2022/discussion/334484)

# 7/22
* Notebook
    * [日本語記載のLGBMNotebook](https://www.kaggle.com/code/wasshoiwasshoi/tps-2022-july-gmm-clustering-with-lightgbm)