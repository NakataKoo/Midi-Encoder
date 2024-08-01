This repository is not official.

## Getting Started

### Environments

仮想環境を構築：

```
python3 -m venv venv
source venv/bin/activate
```

* Python 3.8.8
* Ubuntu 20.04.2 LTS
* Read [requirements.txt](/requirements.txt) for other Python libraries

```pip install -r requirements.txt```を実行後、以下を実行:

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchdata==0.4.0
```

Pytorchの最新バージョンだと、torchtextが使えないらしい。

> 参考：https://github.com/pytorch/serve/issues/3066

以下のリンクを参考に、Pytorch,CUDA,torchtextのバージョンを合わせる

- https://pypi.org/project/torchtext/
- https://pytorch.org/get-started/previous-versions/

研究室のA40サーバー（CUDA 12.1）では、以下でも上手くいった(torch=2.2.0, CUDA=12.1, torchtext=0.17.0)

```
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install torchtext==0.17.0
```


### Data Download

ルートディレクトリにて、以下を実行

```
mkdir data
cd data
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
tar -zxvf lmd_full.tar.gz
rm lmd_full.tar.gz
```

* [Lakh MIDI Dataset (LMD-full)](https://colinraffel.com/projects/lmd/)

### Data Preprocess

ルートディレクトリに戻り以下を実行（1時間30分以上かかるかもしれない）:

[convert_remi.py](/convert_remi.py) is to obtain the bar-level REMI+ representations from LMD. It supports the parallel processing by specifying the number of processes to --num_process argument.
```
python convert_remi.py --src_path "./data/lmd_full/" --dest_path "./data/lmd_full_remi/" --num_process 100
```

合計900万件以上のデータを取得できる

### Model Training
You should modify [config.json](/config.json) for mode change ("BERT", "BERT-aug", "BERT-neighbor", "BERT-dropout"). By setting "strategy" (ex. ddp) in [train.py](/train.py) and "gpus" in [config.json](/config.json) (ex. [0, 1, 2]), you can train the models with distributed GPU settings of pytorch-lightining. Here is an example of BERT-neighbor configurations.

以下のtrain.pyのpl.Trainerの「strategy」を例えばddpなどと設定し、上記のconfig.jsonの "gpus"(例: [0, 1, 2])を設定することで、pytorch-lightiningの分散GPU設定でモデルを学習することができます。

```json
{
    "random_seed": 0,
    "batch_size": 24,
    "num_workers": 16,
    "dim": 768,
    "depth": 12,
    "heads": 12,
    "max_len": 512,
    "rate": 0,
    "masking": 0.8,
    "replace": 0.1,
    "loss_weights": [1, 0.1],
    "lr": 1e-4,
    "epochs": 3,
    "warm_up": 10000,
    "temp": 0.1,
    "gpus": [0],
    "mode": "BERT-neighbor"
}
```


For training the BERT-variants models, the command is as below;
```
python train.py
```

### Model Inference
You can obtain seven evaluation metrics (chords, groove patterns, instruments, tempo, mean velocity, mean duration, song clustering) from [test.ipynb](/test.ipynb).


## Appreciation
I have learned a lot and reused available codes from [dvruette FIGARO](https://github.com/dvruette/figaro), [lucidrains vit-pytorch](https://github.com/lucidrains/vit-pytorch), and [sthalles SimCLR](https://github.com/sthalles/SimCLR/blob/master/simclr.py). Also, I have applied [gautierdag noam scheduler](https://gist.github.com/gautierdag/925760d4295080c1860259dba43e4c01) for learning warm-up, and positional encodings from [dreamgonfly transformer-pytorch](https://github.com/dreamgonfly/transformer-pytorch/blob/master/embeddings.py).


## References
Sangjun Han, Hyeongrae Ihm, Woohyung Lim (LG AI Research), "Systematic Analysis of Music Representations from BERT"