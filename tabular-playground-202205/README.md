# 2022/05
多くの特徴量を含む2値分類問題。
33個のカラム(特徴)がある。
![](2022-05-09-23-30-14.png)

# log
# 5/9
* pandas profileを使ってデータの調査
* どんなデータが与えられているか確認

# 5/10
* Get Started EDA
* https://www.kaggle.com/code/calebreigada/getting-started-eda-preprocessing

# 5/11
* 日本語のEDA
* PyTorchを使ったニューラルネットワーク
* https://www.kaggle.com/code/aboriginal3153/ver-tps-mar-22-neural-network-with-pytorch

# 5/12
* LightGBMの初回submit
* https://www.kaggle.com/code/hikarumoriya/simple-lightgbm
* ![](2022-05-12-21-46-42.png)
* Score: 0.9299 (LB Score: 0.99830)
* こちらもニューラルネットワークだけど分かりやすいNoteBook
    * https://www.kaggle.com/code/kellibelcher/tps-may-2022-eda-lgbm-neural-networks
* matplotlibでf_00からf_30までのデータをヒストグラム化

 # 5/15
 * matplotlibでf_00からf_30までのデータをヒストグラム化する
   * plot.showが処理遅くてグラフがプロットできない
     * 描画に時間がかかる？

# 5/16
### 今日のdiscussion
* トップ3の特徴量エンジニア

Interaction vs Correlationのトピックで、@wti200は、特徴空間のある投影が、ターゲット確率の異なる3つの領域に分割されることを示た。これらの図から、特徴の相互作用を導き出すことができる。特に、3つの投影が有効である。

* f_02とf_21への射影
* f_05とf_22への射影
* f_00+f_01への投影とf_26への投影

# 5/17
### 今日のdiscussion
* a little finding about f_27

有益な情報かどうかわからないが配列を一通り見てP、Q、R、S、Tの文字が常に配列の8文字目にあることがわかった。しかも、それらの文字が最後の位置に配置されることはないらしい。

* plot.showが遅い理由(?)
  * 単純にプロット数が多くて間に合っていなかった
  * f_27カラムがカテゴリ変数だったのでグラフ描画できなかった
  * `fig, ax = plt.subplots(5, 6)`として`ax[0, 0], ax[0, 1]`のように描画するindex番号を割り振る必要があった
    * ![](2022-05-17-15-20-05.png)

* これが一番分かりやすいEDAだったので解読していく
  * https://www.kaggle.com/code/calebreigada/getting-started-eda-preprocessing

# 5/18
* pandas100本ノック(50/100)

# 5/19
* pandas100本ノック