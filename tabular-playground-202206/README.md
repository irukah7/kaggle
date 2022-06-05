# Tabular Playground Series - Jun 2022
6月のPlaygroundシリーズは全てデータインピュテーション(代入法)についてです。代入法はデータ欠損値を代入して置き換える方法です。  
このデータセットは5月のPlaygroundに似ているが`target`がなくなっています。また、データセットには欠損値があるので、これらの値がどうあるべきか予測することです。
![](2022-06-01-17-54-02.png)

* ノートブックの目的
    * データのインピュテーション(代入)

データセットが示唆するように、データには欠損値があり何らかの方法で欠損データをインピュテーションしなければならない。  
データセット内の欠損データを処理する方法はいくつかあり、簡単なインピュテーション方法を紹介したノートブックがたくさんある。
# Log

# 6/3

* データインピュテーション方法を説明したNotebook
    * https://www.kaggle.com/code/shtrausslearning/tps-model-based-imputation

このノートブックでは、異なるモデルのアンサンブルを含むモデルベースのインピュテーションのアプローチを見ていく。  
教師なし学習と教師あり学習の対象的なモデルを使用する。  
* データインピュテーションの種類
    (続きは後ほど記載)


# 6/4
* Discussion
    * https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/discussion/328568
    * 様々なインピュテーションのテクニック
        * null値のある行を削除する
        * null度が高い特徴(カラム)を削除する
        * null値を平均値・中央値・その他の統計量に置換する
        * 半教師あり学習: 欠損データ代入問題の再定式化
        * 最尤推定量法
        * 多重代入法
        * ファンシーインピュテーション：MICE

# 6/5
* Disucussion
    * https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/discussion/328369
    * 今回の課題はどのようなものか、どんな解決策が挙げられるか、コメントを通して探していくdiscussion