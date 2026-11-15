# vtsimnx 計算チュートリアル（18教材）

旧版vtsim・vtsimnxの計算例から代表的な18例を選び、vtsimnxの検証版で実行できる形に整理しました。同じ計算は統合し、換気・熱・湿気・空気質・空調の異なる計算方法を学べる構成にしています。各ノートには目的、入力、計算コード、検証時の表・図、発展課題があります。

最初に [実行ガイド](00_%E5%AE%9F%E8%A1%8C%E3%82%AC%E3%82%A4%E3%83%89.ipynb) を開いてください。01〜17は計算入力をノート内に持ち、18の外部入力は同じフォルダの `house_annual_input.json.gz` です。

## 教材一覧

GitHubでは計算済みのノートをそのまま閲覧できます。Colabで再計算する場合も、17以外は接続可能なengineが必要です。

| 番号 | 教材 | 主な計算 | Colab |
|---|---|---|---|
| 01 | [三節点の換気計算](01_%E4%B8%89%E7%AF%80%E7%82%B9%E3%81%AE%E6%8F%9B%E6%B0%97%E8%A8%88%E7%AE%97.ipynb) | 固定風量100 m³/hと2つの開口から、室内圧力と圧力損失を求める。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/01_%E4%B8%89%E7%AF%80%E7%82%B9%E3%81%AE%E6%8F%9B%E6%B0%97%E8%A8%88%E7%AE%97.ipynb) |
| 02 | [風圧と隙間](02_%E9%A2%A8%E5%9C%A7%E3%81%A8%E9%9A%99%E9%96%93.ipynb) | 風上・風下の風圧と隙間のべき乗則から流量・室内圧力を求める。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/02_%E9%A2%A8%E5%9C%A7%E3%81%A8%E9%9A%99%E9%96%93.ipynb) |
| 03 | [ファンのPQ特性](03_%E3%83%95%E3%82%A1%E3%83%B3%E3%81%AEPQ%E7%89%B9%E6%80%A7.ipynb) | ファンPQ特性と流路抵抗から運転点を求める。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/03_%E3%83%95%E3%82%A1%E3%83%B3%E3%81%AEPQ%E7%89%B9%E6%80%A7.ipynb) |
| 04 | [一室の発熱と蓄熱](04_%E4%B8%80%E5%AE%A4%E3%81%AE%E7%99%BA%E7%86%B1%E3%81%A8%E8%93%84%E7%86%B1.ipynb) | 発熱・換気・壁の熱損失と熱容量から室温の時間変化を求める。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/04_%E4%B8%80%E5%AE%A4%E3%81%AE%E7%99%BA%E7%86%B1%E3%81%A8%E8%93%84%E7%86%B1.ipynb) |
| 05 | [授業の9節点熱回路](05_%E6%8E%88%E6%A5%AD%E3%81%AE9%E7%AF%80%E7%82%B9%E7%86%B1%E5%9B%9E%E8%B7%AF.ipynb) | 4種類の壁を9節点の熱回路で表し、48時間の温度応答を求める。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/05_%E6%8E%88%E6%A5%AD%E3%81%AE9%E7%AF%80%E7%82%B9%E7%86%B1%E5%9B%9E%E8%B7%AF.ipynb) |
| 06 | [壁層・日射・夜間放射](06_%E5%A3%81%E5%B1%A4%E3%83%BB%E6%97%A5%E5%B0%84%E3%83%BB%E5%A4%9C%E9%96%93%E6%94%BE%E5%B0%84.ipynb) | 壁層のRC法と応答係数法、日射・夜間放射の影響を比較する。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/06_%E5%A3%81%E5%B1%A4%E3%83%BB%E6%97%A5%E5%B0%84%E3%83%BB%E5%A4%9C%E9%96%93%E6%94%BE%E5%B0%84.ipynb) |
| 07 | [湿気容量と吸放湿](07_%E6%B9%BF%E6%B0%97%E5%AE%B9%E9%87%8F%E3%81%A8%E5%90%B8%E6%94%BE%E6%B9%BF.ipynb) | 湿気容量の有無による絶対湿度の応答を比較する。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/07_%E6%B9%BF%E6%B0%97%E5%AE%B9%E9%87%8F%E3%81%A8%E5%90%B8%E6%94%BE%E6%B9%BF.ipynb) |
| 08 | [CO2減衰と多室保存](08_CO2%E6%B8%9B%E8%A1%B0%E3%81%A8%E5%A4%9A%E5%AE%A4%E4%BF%9D%E5%AD%98.ipynb) | CO₂濃度の減衰と多室間の移流・保存を確認する。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/08_CO2%E6%B8%9B%E8%A1%B0%E3%81%A8%E5%A4%9A%E5%AE%A4%E4%BF%9D%E5%AD%98.ipynb) |
| 09 | [粉塵沈着とフィルター](09_%E7%B2%89%E5%A1%B5%E6%B2%88%E7%9D%80%E3%81%A8%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF%E3%83%BC.ipynb) | 粒子濃度に対する沈着とフィルターの影響を比較する。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/09_%E7%B2%89%E5%A1%B5%E6%B2%88%E7%9D%80%E3%81%A8%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF%E3%83%BC.ipynb) |
| 10 | [床下を介した多室暖房](10_%E5%BA%8A%E4%B8%8B%E3%82%92%E4%BB%8B%E3%81%97%E3%81%9F%E5%A4%9A%E5%AE%A4%E6%9A%96%E6%88%BF.ipynb) | 床下から各室への暖気分配と室温を求める縮約モデル。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/10_%E5%BA%8A%E4%B8%8B%E3%82%92%E4%BB%8B%E3%81%97%E3%81%9F%E5%A4%9A%E5%AE%A4%E6%9A%96%E6%88%BF.ipynb) |
| 11 | [顕熱交換と全熱交換](11_%E9%A1%95%E7%86%B1%E4%BA%A4%E6%8F%9B%E3%81%A8%E5%85%A8%E7%86%B1%E4%BA%A4%E6%8F%9B.ipynb) | 顕熱交換と全熱交換による給気温湿度を比較する。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/11_%E9%A1%95%E7%86%B1%E4%BA%A4%E6%8F%9B%E3%81%A8%E5%85%A8%E7%86%B1%E4%BA%A4%E6%8F%9B.ipynb) |
| 12 | [別室設置の空調制御](12_%E5%88%A5%E5%AE%A4%E8%A8%AD%E7%BD%AE%E3%81%AE%E7%A9%BA%E8%AA%BF%E5%88%B6%E5%BE%A1.ipynb) | 制御対象の室と空調の吸込・吹出節点を分けて計算する。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/12_%E5%88%A5%E5%AE%A4%E8%A8%AD%E7%BD%AE%E3%81%AE%E7%A9%BA%E8%AA%BF%E5%88%B6%E5%BE%A1.ipynb) |
| 13 | [温度優先・湿度優先・再熱](13_%E6%B8%A9%E5%BA%A6%E5%84%AA%E5%85%88%E3%83%BB%E6%B9%BF%E5%BA%A6%E5%84%AA%E5%85%88%E3%83%BB%E5%86%8D%E7%86%B1.ipynb) | 温度優先・湿度優先・再熱による室温湿度と負荷を比較する。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/13_%E6%B8%A9%E5%BA%A6%E5%84%AA%E5%85%88%E3%83%BB%E6%B9%BF%E5%BA%A6%E5%84%AA%E5%85%88%E3%83%BB%E5%86%8D%E7%86%B1.ipynb) |
| 14 | [潜熱モデル比較](14_%E6%BD%9C%E7%86%B1%E3%83%A2%E3%83%87%E3%83%AB%E6%AF%94%E8%BC%83.ipynb) | rh95・bf・coil_aoafによる潜熱処理の差を比較する。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/14_%E6%BD%9C%E7%86%B1%E3%83%A2%E3%83%87%E3%83%AB%E6%AF%94%E8%BC%83.ipynb) |
| 15 | [隙間倍率6条件の比較](15_%E9%9A%99%E9%96%93%E5%80%8D%E7%8E%876%E6%9D%A1%E4%BB%B6%E3%81%AE%E6%AF%94%E8%BC%83.ipynb) | 隙間倍率0、0.5、1、2、4、8の換気量を比較する。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/15_%E9%9A%99%E9%96%93%E5%80%8D%E7%8E%876%E6%9D%A1%E4%BB%B6%E3%81%AE%E6%AF%94%E8%BC%83.ipynb) |
| 16 | [枚方のαA推定モデル](16_%E6%9E%9A%E6%96%B9%E3%81%AE%CE%B1A%E6%8E%A8%E5%AE%9A%E3%83%A2%E3%83%87%E3%83%AB.ipynb) | 14節点・20流路の初期開口モデルで目標風量との差を確認する。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/16_%E6%9E%9A%E6%96%B9%E3%81%AE%CE%B1A%E6%8E%A8%E5%AE%9A%E3%83%A2%E3%83%87%E3%83%AB.ipynb) |
| 17 | [実測と気象の前処理](17_%E5%AE%9F%E6%B8%AC%E3%81%A8%E6%B0%97%E8%B1%A1%E3%81%AE%E5%89%8D%E5%87%A6%E7%90%86.ipynb) | 合成データで時刻・欠測・集計・湿り空気の物性変換を確認する。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/17_%E5%AE%9F%E6%B8%AC%E3%81%A8%E6%B0%97%E8%B1%A1%E3%81%AE%E5%89%8D%E5%87%A6%E7%90%86.ipynb) |
| 18 | [全館空調の保存入力を再計算](18_%E5%85%A8%E9%A4%A8%E7%A9%BA%E8%AA%BF%E3%81%AE%E4%BF%9D%E5%AD%98%E5%85%A5%E5%8A%9B%E3%82%92%E5%86%8D%E8%A8%88%E7%AE%97.ipynb) | 保存済み年間入力から冬・夏の各24時間を切り出して全館空調を計算する。 | [開く](https://colab.research.google.com/github/iguchi-lab/vtsimnx/blob/docs/tutorials-18-20261115/tutorials/18_%E5%85%A8%E9%A4%A8%E7%A9%BA%E8%AA%BF%E3%81%AE%E4%BF%9D%E5%AD%98%E5%85%A5%E5%8A%9B%E3%82%92%E5%86%8D%E8%A8%88%E7%AE%97.ipynb) |

## ローカルで実行する

1. リポジトリをcloneまたはDownload ZIPで取得します。このPRの確認中は `docs/tutorials-18-20261115` ブランチを取得してください。
2. Python 3.10以上の環境を用意し、`tutorials/` に移動します。
3. 必要なライブラリを導入してJupyterを起動します。

```bash
python -m pip install --upgrade pip "setuptools>=68,<81" wheel
python -m pip install -r requirements.txt
python -m jupyterlab
```

教材17以外はvtsimnx engineを別途起動します。[engineの構築・起動手順](https://github.com/iguchi-lab/vtsimnx/blob/5eeffb2f8012a2fe41bca705450b0dd0a952326d/engine/RUN_FASTAPI.md) に従い、下記の検証commitにクライアントとengineをそろえてください。クライアントのpip導入だけではengineは起動しません。

| 設定 | 既定値・意味 |
|---|---|
| `VTSIMNX_API_URL` | `http://127.0.0.1:8000`。ノートの `BASE_URL` でも変更できます。 |
| `VTSIMNX_API_KEY` | 認証が必要なengineのAPIキー。環境変数やColab Secretsから渡します。 |
| `VTSIMNX_TUTORIAL_DATA` | 作業フォルダ。別の場所から起動する場合は `tutorials/` の絶対パスを指定します。 |

結果は `results/教材番号/` に保存され、Gitの管理対象から除外されます。各ノートの導入セルを上から順に実行してください。

## Colabで実行する

一覧の「開く」からノートを開きます。`BASE_URL` はColabから到達できるengineへ変更してください。Colabの `127.0.0.1` はColab自身を指します。認証が必要な場合は、実行ガイドのColab Secretsの例を利用できます。

教材18は入力が見つからないときだけGitHubの固定commitから約5MBのgzを取得します。保存前にSHA-256を確認します。手動で配置した同梱入力もハッシュを確認します。元の実測CSV・気象CSV・風圧係数CSVやDriveのマウントは不要です。

## 検証した範囲

- 検証日: 2026年11月15日。
- engine / Python client: `5eeffb2f8012a2fe41bca705450b0dd0a952326d`。最新mainを自動追従する設定ではありません。
- Drive配布版の18教材・91コードセルをJupyter nbclientで実行し、実際のHTTP APIによる計算・結果取得と31図の出力を確認しました。依存関係は専用環境へ一度導入し、セルごとの再インストールだけ省略しました。
- GitHub版の計算セルは同じです。教材18の入力取得セルと案内文を変更し、ファイル同梱・取得・ハッシュ検査を別途確認しています。詳細は [元の実行記録](notebook_validation.json) と [GitHub版の確認記録](github_package_validation.json) にあります。
- 教材16は初期開口0.01 m²でのモデル計算です。開口最適化の再現は含みません。
- 教材18は1月15日・8月1日の各24時間、隙間倍率1、rh95です。年間18条件の収束・原典の年間結果との一致は未検証です。短期間の計算は元の初期値から開始するため、年間計算の同日結果とは一致するとは限りません。
- 教材15の隙間倍率0は、全流量固定0で未知圧力がない境界ケースとして表示します。

## 原典と入力データ

各教材末尾には原典名と調査時のS番号を残しています。原典の全ファイル台帳と注釈版は管理者のDrive整理資料に保管しています。このフォルダは代表教材の配布単位です。

教材18の入力は25節点・405換気枝・183面・空調1台、2026年の8760時間を含む保存済みモデルです。気象・風圧・スケジュールを含みます。来歴は [house_input_provenance.json](house_input_provenance.json)、ファイル構成とSHA-256は [input_manifest.json](input_manifest.json) に記録しています。

旧版の実行結果をそのまま再現する範囲と、学習用に縮約した範囲は各教材で区別しています。倍率からC値への換算、時刻の意味、単位、助走期間などの条件も各教材の説明を確認してください。
