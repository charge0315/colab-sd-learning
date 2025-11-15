````markdown
# Stable Diffusion 3.5 差分学習（LoRA） & バッチ画像生成 Notebook

このリポジトリは、**Google Colab 上で Stable Diffusion 3.5 を差分学習（LoRA など）し、学習済み差分モデルを使ってバッチ画像生成を行うためのノートブック**用 README です。

- 学習データ：Google Drive から読み込み  
- 学習方式：ベースモデル + 差分（LoRA / LoRA 互換）  
- 保存先：学習済み差分モデルを Google Drive に保存  
- 画像生成：ポジティブ / ネガティブプロンプトを指定してバッチ生成  

※この README は、`sd35_lora_train_and_infer.ipynb` という Colab ノートブックを想定して書かれています。ファイル名は環境に合わせて読み替えてください。

---

## 1. 前提条件

- Google アカウント（Google Drive, Google Colab が利用可能）
- Google Colab（GPU 付きランタイム）
  - 推奨：T4 / L4 / A100 などの GPU ランタイム
- Stable Diffusion / 画像生成の基礎知識（プロンプト、シード値など）

---

## 2. ノートブック全体構成

ノートブックはおおまかに次のセクションに分かれています。

1. **環境セットアップ**
2. **Google Drive のマウント & パス設定**
3. **ベースモデル（SD 3.5）のダウンロード / 読み込み**
4. **学習データセットの準備（Google Drive）**
5. **差分学習（LoRA トレーニング）**
6. **学習済み差分モデルの保存（Google Drive）**
7. **画像生成用パイプラインの構築**
8. **ポジティブ / ネガティブプロンプトを使ったバッチ画像生成**
9. **オプション：推論パラメータの調整・再現性確保**

---

## 3. 利用開始手順（ざっくり）

1. ノートブックを Google Colab で開く  
2. ランタイムタイプを確認  
   - メニューから **「ランタイム」→「ランタイムのタイプを変更」**  
   - ハードウェア アクセラレータ：`GPU` を選択  
3. セクション順に上から順番にセルを実行していく  
4. 学習データのパス、保存先フォルダ、バッチ生成設定などを自分の環境に合わせて修正  
5. 学習が完了したら、下部の「バッチ画像生成」セクションでプロンプトをまとめて投入して画像生成

---

## 4. Google Drive のマウント & ディレクトリ構成

### 4.1 Drive マウント

ノートブック内の「Google Drive のマウント」セルで次を実行します。

- Google アカウント認証
- `/content/drive/MyDrive` がマウントされる

### 4.2 想定ディレクトリ構成（例）

Google Drive 側で、以下のような構成を想定しています。

```text
MyDrive/
  sd35_project/
    datasets/
      my_dataset_01/
        images/
          img_0001.jpg
          img_0002.png
          ...
        captions/           # 任意
          img_0001.txt
          img_0002.txt
          ...
    models/
      base/
        sd35_base/          # （任意：事前に配置する場合）
      lora/
        sd35_lora_exp001/   # 学習済み差分モデル保存先
    outputs/
      samples/
        ...                 # 生成画像の保存先
      logs/
        ...                 # 学習ログなど（任意）
````

ノートブック内の変数例：

```python
PROJECT_ROOT = "/content/drive/MyDrive/sd35_project"
DATASET_DIR = f"{PROJECT_ROOT}/datasets/my_dataset_01"
LORA_OUTPUT_DIR = f"{PROJECT_ROOT}/models/lora/sd35_lora_exp001"
GENERATED_DIR = f"{PROJECT_ROOT}/outputs/samples"
```

必要に応じて自分の Drive のパスに変更してください。

---

## 5. ベースモデル（Stable Diffusion 3.5）の設定

ノートブックには以下の方針で実装されています。

* Hugging Face Hub から SD 3.5 のベースモデルを取得（例）
* あるいは、すでに Drive にダウンロード済みのモデルを読み込む

典型的には以下のようなパラメータをセル内で指定します。

```python
BASE_MODEL_ID = "stabilityai/stable-diffusion-3.5-large"  # 例：実際のIDに合わせて変更
USE_LOCAL_BASE = False  # True の場合、Drive から読み込み
```

> **注意**
> 実際に利用するモデルIDは、利用規約やライセンス、Colab 環境での互換性を確認した上で設定してください。

---

## 6. 学習データセットの準備

### 6.1 画像とキャプション

差分学習用のデータとして以下を想定します。

* `images/` ディレクトリ内に学習用画像ファイル（jpg / png など）
* 任意で `captions/` ディレクトリに、画像と同名の `.txt` キャプションファイル

  * `img_0001.jpg` → `captions/img_0001.txt`
  * キャプションには、その画像を説明するテキスト（プロンプト）を記述

キャプションファイルがない場合、ノートブック側で簡易的な共通プロンプトを付与する実装も可能です（そこはノートブックの実装に依存します）。

### 6.2 設定セル

ノートブックの「データセット設定」セルで以下のような変数を設定します。

```python
IMAGE_DIR = f"{DATASET_DIR}/images"
CAPTION_DIR = f"{DATASET_DIR}/captions"   # キャプション利用しない場合は None でも可
RESOLUTION = 768                          # 例：学習時の解像度
```

---

## 7. 差分学習（LoRA トレーニング）

### 7.1 トレーニング設定

「学習設定」セルでは、以下のようなハイパーパラメータを指定します。

```python
LORA_RANK = 16
LORA_ALPHA = 32
TRAIN_BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 1
MAX_TRAIN_STEPS = 1000  # ステップ数を直接指定する場合
SAVE_STEPS = 100        # 何ステップごとに保存するか
LORA_OUTPUT_DIR = f"{PROJECT_ROOT}/models/lora/sd35_lora_exp001"
```

### 7.2 実行

「学習実行」セルを実行すると、以下が行われます。

* ベースモデル + LoRA の差分層を構築
* 画像 + キャプションを用いたファインチューニング
* 指定ステップごとに中間チェックポイントを Drive に保存
* 学習完了時に最終的な LoRA 差分モデルを Drive に保存

保存形式は、ノートブックの実装によって異なりますが、以下のような想定です。

* Diffusers 形式の LoRA 重み (`pytorch_lora_weights.safetensors` 等)
* もしくは LoRA 用ディレクトリ一式

---

## 8. 学習済み差分モデルの保存先

学習完了後、以下のようなパスに保存されます（例）：

```text
MyDrive/sd35_project/models/lora/sd35_lora_exp001/
  - adapter_config.json
  - adapter_model.safetensors
  - もしくは diffusers 形式の各種ファイル
```

学習をやり直したい場合は、`sd35_lora_exp002` のようにディレクトリを変えて実験を分離すると管理しやすくなります。

---

## 9. 画像生成（バッチ処理）

### 9.1 生成用の基本フロー

ノートブックの後半では、

1. ベースモデル（SD 3.5）をロード
2. 指定した LoRA 差分モデルをマージ or アタッチ
3. 画像生成用パイプライン（例：Diffusers の `StableDiffusionPipeline`）を構築
4. 指定されたプロンプトリストに対して画像生成を一括実行
5. 生成画像を Google Drive の `outputs/samples` に保存

という流れになっています。

### 9.2 ポジティブ / ネガティブプロンプトの入力

バッチ生成用に、以下のような形式のテキストファイルを想定しています。

```text
# prompts_batch.txt の例（CSV っぽく 1 行 1 ジョブ）
positive prompt 1|||negative prompt 1
positive prompt 2|||negative prompt 2
positive prompt 3|||negative prompt 3
...
```

* 区切り文字 `|||` によって、ポジティブプロンプトとネガティブプロンプトを分離
* 1 行が 1 生成ジョブに対応

ノートブック内の設定例：

```python
PROMPT_FILE = f"{PROJECT_ROOT}/prompts/prompts_batch.txt"
NUM_IMAGES_PER_PROMPT = 2
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 30
OUTPUT_DIR = GENERATED_DIR
SEED = 42  # 再現性を確保したい場合に固定
```

### 9.3 バッチ生成セルの挙動

「バッチ画像生成」セルを実行すると：

* `PROMPT_FILE` を 1 行ずつ読み込み
* 行ごとに `positive`, `negative` に分割
* `NUM_IMAGES_PER_PROMPT` 枚ずつ生成
* ファイル名例：

  * `prompt000_pos.png`
  * `prompt000_pos_1.png`
  * `prompt001_pos.png` …など

生成結果は、Google Drive の `outputs/samples/` 配下に保存されます。

---

## 10. よくある調整ポイント

* **学習が遅い / メモリ不足になる**

  * `RESOLUTION` を下げる（例：768 → 512）
  * `TRAIN_BATCH_SIZE` を減らす
  * `GRADIENT_ACCUMULATION_STEPS` を増やす
* **結果が思った通りにならない**

  * 学習ステップ数を増やす (`NUM_EPOCHS` / `MAX_TRAIN_STEPS`)
  * 学習データの枚数を増やす
  * キャプションの質を見直す
* **生成結果が全体的に崩れる**

  * 学習率を下げる（`1e-4` → `5e-5` など）
  * LoRA の rank / alpha を見直す（過学習・表現力のバランス）

---

## 11. トラブルシューティング

* **`CUDA out of memory` エラーが出る**

  * 解像度・バッチサイズを下げる
  * 他の GPU プロセスがいないか確認（Colab を再起動）
* **ライブラリのバージョンでエラー**

  * ノートブック先頭の「依存ライブラリインストール」セルを再実行
  * バージョン指定（`diffusers==X.Y.Z` など）を見直す
* **Drive 内パスが見つからない**

  * `/content/drive/MyDrive` が正しいか再確認
  * パス文字列のタイプミス・全角/半角・スペースに注意

---

## 12. まとめ

このノートブック & README は、

* Google Colab + Google Drive だけで
* Stable Diffusion 3.5 の差分学習（LoRA）
* 学習済み差分モデルの Drive 保存
* ポジ/ネガプロンプトを使ったバッチ画像生成

までを一気通貫で回せることを目標にしています。

あとは、
**「どんなデータで、どんなプロンプトを入れるか」** というクリエイティブ勝負の世界です。
パラメータをガチャガチャ変えながら、自分だけの SD3.5 を育てていきましょう 🎨🧠

```
```
