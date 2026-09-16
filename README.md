# speck-tutorial

SynSense Speck2f Dev Kitの接続確認、DVS記録、モデルの配置、N-MNISTを使った学習・実機推論のサンプルです。

## プログラム一覧

| 番号 | プログラム | 内容 | Speck実機 |
| --- | --- | --- | --- |
| 01 | [01_check_device.py](script/01_check_device.py) | ライブラリのバージョンと接続デバイスを表示 | 接続確認には必要 |
| 02 | [02_record_dvs.py](script/02_record_dvs.py) | DVSイベントを記録し、停止後にMP4へ変換 | 必要 |
| 03 | [03_deploy_toy_model.py](script/03_deploy_toy_model.py) | 小さなCNNをSNNへ変換し、Speckへ配置 | 必要 |
| 04-01 | [04_01_NMNIST_ann_train.py](script/04_01_NMNIST_ann_train.py) | N-MNISTでANNを学習・評価 | 不要 |
| 04-02 | [04_02_NMNIST_snn_train.py](script/04_02_NMNIST_snn_train.py) | N-MNISTでSNNを直接学習・評価 | 不要 |
| 05 | [05_NMNIST_speck_inference.py](script/05_NMNIST_speck_inference.py) | 学習済みモデルをSpeckへ配置してN-MNISTを推論 | 必要（dry-runは不要） |

01で接続を確認してから02または03を実行してください。04の2本は独立した学習プログラムで、02の記録データや03のモデルは使用しません。04-02の実行前に04-01を学習させる必要もありません。

05では04-01または04-02が保存したチェックポイントを使用します。実行前に01で接続を確認し、同じSpeckを使う02・03などのプログラムは終了してください。

## 共通の準備

以下のコマンドはリポジトリのルート（`README.md`があるディレクトリ）から実行します。`output/`や`data/`などの相対パスも、実行時のディレクトリが基準です。

### ホスト側でコンテナを準備する

Apptainerが利用できるLinuxホストで、次を実行します。

```bash
apptainer build --fakeroot apptainer/speck.sif apptainer/speck.def
apptainer test apptainer/speck.sif
```

既存のSIFに必要なライブラリが揃っていれば、そのファイルを使用できます。02にはOpenCV、04・05にはTonicが必要です（04はtqdmも使用します）。古いSIFを使用している場合は、現在の定義から再ビルドしてください（04の節に別名でビルドする例もあります）。

USB権限の初回設定とビルドの詳細は[Apptainer環境のREADME](apptainer/README.md)を参照してください。01〜03・05ではSpeckをUSB 3対応ケーブルでUSB 3ポートへ接続します。

### コンテナに入る

ホスト側で次を実行します。

```bash
apptainer shell apptainer/speck.sif
```

以降の`python script/...`は、このコンテナ内で実行します。終了するときは`exit`を入力してください。

`apptainer/run_cnt.sh`も利用できますが、SIFの場所と`/ldisk`へのbindが特定の環境向けに固定されています。利用する場合は自分の保存先に合わせてから、ホスト側で次を実行してください。

```bash
sh apptainer/run_cnt.sh
```

SpeckにはApptainer標準の`/dev`共有を使用します。`--contain`や明示的な`--bind /dev/bus/usb:/dev/bus/usb`は追加しないでください。この環境では明示bindに`nodev`が付き、`lsusb`には表示されてもSamnaから開けなくなることを確認しています。

## 01: 接続デバイスの確認

コンテナ内で実行します。

```bash
python script/01_check_device.py
```

Samna・Sinabsのバージョン情報と、`sio.get_device_map()`で得たデバイス一覧を表示します。Speck2f Dev Kitが1台接続されていれば、通常は`speck2fdevkit:0`に対応する情報が表示されます。バージョンが`unknown`でも、それだけで接続失敗を意味するわけではありません。

デバイスが見つからない場合は、USB接続、ホスト側のudev設定、起動時のbind指定を確認してください。02と03は同じSpeckを同時に使用せず、一方の終了後に実行します。

## 02: DVSイベントを記録して動画にする

```bash
python script/02_record_dvs.py
```

記録開始のメッセージが出たらSpeckの前で手や物体を動かします。ターミナルで`q`を押すと記録を停止し、MP4への変換を開始します。Enterは不要です。録画中の`Ctrl-C`も停止・動画変換として扱われます。

標準の出力は`output/dvs_recording.mp4`（128×128、30 FPS）です。極性0のイベントを緑、極性1を赤で描画します。これは明るさの変化を記録したイベントの可視化で、通常のカメラ映像とは見え方が異なります。

保存先や明るさを変更する例:

```bash
python script/02_record_dvs.py \
  --output output/example.mp4 \
  --fps 30 \
  --gain 64 \
  --stop-key q
```

5秒で自動停止する例（キー入力のできない実行環境でも利用可能）:

```bash
python script/02_record_dvs.py --duration 5 --output output/dvs_5s.mp4
```

| オプション | 初期値 | 内容 |
| --- | --- | --- |
| `--device` | `speck2fdevkit:0` | 使用するデバイス |
| `--output` | `output/dvs_recording.mp4` | 動画の保存先 |
| `--fps` | `30` | 動画のフレームレート |
| `--gain` | `32` | 1イベントあたりの描画輝度。大きくすると明るくなる |
| `--stop-key` | `q` | 停止キー（1文字、大文字・小文字は区別しない） |
| `--poll-interval` | `0.01` | イベント回収間隔（秒） |
| `--duration` | 指定なし | 自動停止までの秒数。指定時は停止キー入力を使用しない |
| `--codec` | `mp4v` | OpenCVの4文字コーデック指定 |

```bash
python script/02_record_dvs.py --help
```

録画中は出力先ディレクトリに一時イベントファイルを作成し、停止後に順次動画化します。ディスクの空き容量を確保してください。成功時は一時ファイルを削除し、失敗してイベントが残っている場合はその保存先を表示します。同名の動画は上書きされるため、録画を残す場合は`--output`を変えてください。

## 03: 小さなモデルをSpeckへ配置する

```bash
python script/03_deploy_toy_model.py
```

プログラム内でCNNを作成し、Sinabsの`from_model()`でSNNへ変換した後、`DynapcnnNetwork`で重みを離散化し、Speckのコアへ自動配置します。成功時は次のメッセージとモデル情報を表示します。

```text
Deployment succeeded.
```

対象はプログラム先頭の`DEVICE = "speck2fdevkit:0"`で指定されています。別のデバイス番号を使う場合は、この値を01の確認結果に合わせて変更してください。03にはコマンドライン引数はありません。

このサンプルは未学習モデルの配置確認です。入力は1チャンネル・28×28を想定し、`dvs_input=False`で構築しています。実行するだけでDVS映像の分類や精度評価が行われるわけではありません。

## 04: N-MNISTを使った学習

04-01と04-02はホストのCPUまたはGPU上で学習します。Speckを接続する必要はありません。

### 実行環境

現在の`apptainer/speck.def`にはTonicとtqdmが含まれています。以前のSIFを使用している場合は、更新した定義から再ビルドしてください。既存のSIFを残す場合は、ホスト側で別名のSIFを作成できます。

```bash
apptainer build --fakeroot apptainer/speck-training.sif apptainer/speck.def
apptainer test apptainer/speck-training.sif
apptainer shell apptainer/speck-training.sif
```

コンテナ内で依存関係を確認できます。

```bash
python -m pip check
python -c 'import tonic, tqdm, torch, sinabs; print("training imports: OK")'
```

デバイスは自動選択されます。CUDAが利用可能なら`cuda:0`、それ以外では`cpu`を使います。現在の定義はCPU版PyTorchなので、コードを編集せずCPUで実行できます。進捗表示にはターミナルにも対応する`tqdm.auto`を使用しています。

GPUで学習する場合は、CUDA対応PyTorchとNVIDIAドライバーに適合する別途の環境を用意し、Apptainerを`--nv`付きで起動します。CPU版SIFに`--nv`を付けるだけではCUDA対応にはなりません。GPU用環境では、次が`True`になることを確認してください。

```bash
python -c 'import torch; print(torch.cuda.is_available())'
```

### 04-01: ANNの学習

コンテナ内で実行します。

```bash
python script/04_01_NMNIST_ann_train.py --data data
```

N-MNISTのイベントをサンプルごとに1フレームへ積算し、正負の極性を2チャンネルとしてCNNを学習します。開始時に変換後データの形状を表示し、各エポックの学習損失とテスト精度（`accuracy`）を表示します。

### 04-02: SNNの直接学習

```bash
python script/04_02_NMNIST_snn_train.py --data data
```

N-MNISTの各サンプルを100時間ステップへ分割し、Sinabsの`IAFSqueeze`を含むSNNをBPTT（時間方向に展開した誤差逆伝播）で学習します。各エポックの学習損失とテスト精度（`BPTT accuracy`）を表示します。独立したサンプル間で膜電位が持ち越されないよう、学習・評価ともバッチごとに状態をリセットします。時間ステップを扱うため、特にCPUでは学習に時間がかかります。

### データ保存先と学習設定

`--data`は両プログラム共通のデータセット保存先で、初期値は`data`です。データ未取得の場合はTonicがダウンロードするため、初回はネットワーク接続と保存先への書き込み権限が必要です。同じ保存先を指定すれば、04-01と04-02でデータを共有できます。

`run_cnt.sh`で`/ldisk`をbindしている場合は、例えば次のように保存先を変更できます。

```bash
python script/04_01_NMNIST_ann_train.py --data /ldisk/habara/data
python script/04_02_NMNIST_snn_train.py --data /ldisk/habara/data
```

両プログラムで以下のオプションを指定できます。

| オプション | 初期値 | 内容 |
| --- | --- | --- |
| `--data` | `data` | データセット保存先 |
| `--device` | `auto` | `auto`、`cpu`、`cuda:0`など |
| `--epochs` | `10` | エポック数（1以上） |
| `--batch-size` | `4` | バッチサイズ（1以上） |
| `--num-workers` | `4` | DataLoaderワーカー数（0でメインプロセスのみ） |
| `--output` | ANN: `output/nmnist_ann.pt`、SNN: `output/nmnist_snn.pt` | 学習済みチェックポイントの保存先 |

CPUで1エポックだけ実行する例:

```bash
python script/04_01_NMNIST_ann_train.py --data data --device cpu --epochs 1 --num-workers 0
python script/04_02_NMNIST_snn_train.py --data data --device cpu --epochs 1 --num-workers 0
```

1エポックでも学習データ全体を処理します。学習率は両方とも`1e-3`、時間ステップ数はANNが1、SNNが100で、変更する場合はコードを編集してください。SNNは固定バッチサイズで処理するため、学習・評価ともバッチサイズに満たない末尾のデータを除外します。ANNは学習時のみ末尾の端数を除外し、評価には全サンプルを使用します。

各エポックの終了時に`--output`へ重み・モデル種別・完了エポック数を保存します。同名ファイルは上書きされるため、過去の重みを残す場合は別の保存先を指定してください。SNNの一時的な膜電位は保存しません。これは05で推論するためのチェックポイントで、optimizerの状態を含む学習再開機能はありません。

### 学習処理の簡易テスト

コンテナ内で次を実行すると、合成イベントを使ってCPUでの学習・評価、SNNの状態リセット、引数の検証を確認できます。Speck実機やN-MNISTのダウンロードは不要です。実データでの学習精度を保証するテストではありません。

```bash
python -m unittest discover -s tests -v
```

## 05: Speck上でN-MNISTを推論する

学習済みモデルをSpeckへ配置し、N-MNISTのテストイベントをUSB経由で再生して、最終層の発火数から数字を判定します。Speck内蔵DVSカメラの入力は無効にします。ホストのCPUはイベントの準備・送受信・集計を担当し、ニューラルネットワークの推論はSpeck上で実行します。GPUは不要です。

### 1. 学習済みチェックポイントを用意する

更新した04のどちらかを実行してください。SNNを直接学習する例:

```bash
python script/04_02_NMNIST_snn_train.py --data data --output output/nmnist_snn.pt
```

ANNを使う場合:

```bash
python script/04_01_NMNIST_ann_train.py --data data --output output/nmnist_ann.pt
```

05はチェックポイントからANN/SNNを自動判別し、ANNの場合はSinabsでSNNへ変換します。その後、両方ともSpeck向けに重みを離散化します。以前の保存機能のない04で学習を終了していた場合は、更新後の04で再学習が必要です。任意のモデルや生の`state_dict`ではなく、このリポジトリの04が保存したファイルを指定してください。

### 2. 実機なしで配置設定を確認する

```bash
python script/05_NMNIST_speck_inference.py --checkpoint output/nmnist_snn.pt --dry-run
```

`Dry run OK`と論理層から物理コアへの割り当てが表示されれば、重みの読み込み・変換・配置設定の検証が成功しています。USBデバイスは開かず、データセットも読み込みません。実機推論や分類精度の検証ではありません。

### 3. Speckで推論する

Speckを接続し、同じ実機を使用している他のプログラムを終了してから実行します。

```bash
python script/01_check_device.py
python script/05_NMNIST_speck_inference.py \
  --checkpoint output/nmnist_snn.pt \
  --data data \
  --num-samples 100 \
  --output output/nmnist_speck.csv
```

ANNの重みを使う場合は`--checkpoint output/nmnist_ann.pt`へ変更します。各サンプルの正解・予測・出力発火数を表示し、最後に`Speck accuracy`と判定不能件数（`undecided`）を表示します。`Ctrl-C`で中断するとデバイスを閉じ、CSVには完了済みのサンプルが残ります。

| オプション | 初期値 | 内容 |
| --- | --- | --- |
| `--checkpoint` | 必須 | 04で保存したチェックポイント |
| `--data` | `data` | N-MNIST保存先。未取得ならテストデータをダウンロード |
| `--device` | `speck2fdevkit:0` | 使用するSpeck |
| `--num-samples` | `100` | 残りのテストデータ全体から等間隔に抽出する件数。`0`で残り全件 |
| `--start-index` | `0` | 抽出対象の先頭インデックス |
| `--output` | 指定なし | 任意のCSV保存先。既存ファイルは上書きしない |
| `--dry-run` | 無効 | 実機に接続せず配置設定のみ検証 |

先頭の連続した少数サンプルだけに評価が偏らないよう、指定範囲全体から等間隔に抽出します。全件評価は`--num-samples 0`、特定の1件だけを試す場合は`--start-index 123 --num-samples 1`と指定します。

最終層の出力がない場合は`prediction=-1, status=no_spikes`、最多発火数が同率の場合は`prediction=-1, status=tie`とし、どちらも正解率の分母に含めて不正解として扱います。CSVにはインデックス、正解、予測、状態、正誤、数字0〜9の発火数を記録します。

各サンプルの前に実機の膜電位をリセットするため、イベント再生時間に加えて待ち時間があります。ANN→SNN変換、重みの離散化、学習時の時間ビン化と実機のイベント処理の違いにより、04のテスト精度と05の精度は一致するとは限りません。

イベントの入力と最終層の発火数による判定は[Sinabs公式N-MNISTチュートリアル](https://sinabs.readthedocs.io/v3.0.4/speck/notebooks/nmnist_quick_start.html)に沿っています。実機なしのテストは、04の節と同じ`python -m unittest discover -s tests -v`で実行できます。
