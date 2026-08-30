# FADLinear

This repository contains the training and evaluation code for FADLinear and the seven comparison models used in the accepted study: DLinear, PatchTST, TimeMixer, iTransformer, WPMixer, FilterTS, and FITS.

## Environment

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the CUDA build of PyTorch appropriate for the target system when GPU execution is required.

## Data

Raw datasets are not distributed in this repository. Place the files in the following locations:

```text
dataset/
  ETT/
    ETTh1.csv
    ETTh2.csv
    ETTm1.csv
    ETTm2.csv
    ETTm2_5min_SAW.csv
  Weather/
    weather.csv
```

The Weather experiments use `T (degC)` as the target column. The data loader moves the selected target to the final column for `MS` forecasting. SAW is an ETTm2-derived exploratory stress dataset and can be generated with:

```bash
python dataset/simulation/generate_ETTm2_5min_SAW.py
```

## Experiments

Launch an experiment directly from the repository root:

```bash
python main.py --is_training 1 --model_id ETTh1_FADLinear --model FADLinear --data ETTh1 --root_path ./dataset/ETT --data_path ETTh1.csv --features MS --target OT --seq_len 336 --pred_len 96 --enc_in 7 --c_out 1 --batch_size 32 --train_epochs 100 --learning_rate 0.001 --itr 3 --random_seed 2026 --freq h
```

## Efficiency Benchmark

The optional efficiency utility accepts a user-provided JSON profile containing model arguments and checkpoint paths:

```bash
python core/exp/benchmark_efficiency.py --config path/to/profile.json --output-dir efficiency_results --device cuda:0
```

Training outputs are written under `experiments_root`. Generated outputs are excluded from version control.
