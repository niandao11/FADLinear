import argparse
import datetime
import hashlib
import json
import random

import numpy as np
import torch

from core.exp.exp_main import Exp_Main


MODEL_NAMES = (
    "FADLinear",
    "DLinear",
    "PatchTST",
    "TimeMixer",
    "itransformer",
    "WPMixer",
    "FilterTS",
    "FITS",
)


def strict_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_parser():
    parser = argparse.ArgumentParser(description="Long-horizon time-series forecasting")
    parser.add_argument("--is_training", type=int, required=True)
    parser.add_argument("--train_only", type=strict_bool, default=False)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=MODEL_NAMES)
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=("ETTh1", "ETTh2", "ETTm1", "ETTm2", "ETTm2_SAW", "weather"),
    )
    parser.add_argument("--root_path", type=str, default="./dataset/ETT")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="MS", choices=("M", "S", "MS"))
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=1)
    parser.add_argument("--individual", type=strict_bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--subtract_last", type=int, default=0)
    parser.add_argument("--e_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--output_attention", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--fc_dropout", type=float, default=0.05)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--padding_patch", type=str, default="end")
    parser.add_argument("--revin", type=int, default=1)
    parser.add_argument("--affine", type=int, default=0)
    parser.add_argument("--decomposition", type=int, default=0)
    parser.add_argument("--kernel_size", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=2026)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--use_norm", type=strict_bool, default=True)
    parser.add_argument("--class_strategy", type=str, default="cls", choices=("cls", "projection"))
    parser.add_argument("--down_sampling_layers", type=int, default=3)
    parser.add_argument("--down_sampling_window", type=int, default=2)
    parser.add_argument("--down_sampling_method", type=str, default="avg", choices=("avg", "max", "conv"))
    parser.add_argument("--channel_independence", type=int, default=1, choices=(0, 1))
    parser.add_argument("--decomp_method", type=str, default="moving_avg", choices=("moving_avg", "dft_decomp"))
    parser.add_argument("--use_future_temporal_feature", type=int, default=0, choices=(0, 1))
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--filter_type", type=str, default="all")
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument("--bandwidth", type=int, default=1)
    parser.add_argument("--top_K_static_freqs", type=int, default=10)
    parser.add_argument("--embedding", type=str, default="fourier_interpolate")
    parser.add_argument("--wavelet", type=str, default="db2")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--no_decomposition", action="store_true")
    parser.add_argument("--tfactor", type=int, default=5)
    parser.add_argument("--dfactor", type=int, default=5)
    parser.add_argument("--embedding_dropout", type=float, default=None)
    parser.add_argument("--fits_h_order", type=int, default=6)
    parser.add_argument("--fits_base_t", type=int, default=24)
    parser.add_argument("--fits_train_mode", type=int, choices=(1, 2), default=1)
    parser.add_argument("--use_gpu", type=strict_bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true")
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--experiment_root", type=str, default="experiments_root")
    return parser


def get_setting(args, repeat_index):
    if args.model == "FITS":
        identity = {
            "root_path": args.root_path,
            "data": args.data,
            "data_path": args.data_path,
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "model": args.model,
            "fits_h_order": args.fits_h_order,
            "fits_base_t": args.fits_base_t,
            "fits_train_mode": args.fits_train_mode,
            "individual": bool(args.individual),
            "current_seed": args.current_seed,
            "repeat_index": repeat_index,
        }
        canonical = json.dumps(identity, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
        return f"{args.data}_FITS_pl{args.pred_len}_{digest}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{args.data}_{args.model}_{args.target}_pl{args.pred_len}_{timestamp}_{repeat_index}"


def configure_device(args):
    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(device_id) for device_id in args.devices.split(",")]
        args.gpu = args.device_ids[0]


def run(args):
    configure_device(args)
    print("Args in experiment:")
    print(args)
    if args.is_training:
        for repeat_index in range(args.itr):
            args.current_seed = args.random_seed + repeat_index
            set_random_seed(args.current_seed)
            setting = get_setting(args, repeat_index)
            experiment = Exp_Main(args)
            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            experiment.train(setting)
            if not args.train_only:
                print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                experiment.test(setting, test=0)
            if args.do_predict:
                print(f">>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                experiment.predict(setting, load=False)
            torch.cuda.empty_cache()
        return
    args.current_seed = args.random_seed
    set_random_seed(args.current_seed)
    setting = get_setting(args, 0)
    experiment = Exp_Main(args)
    if args.do_predict:
        print(f">>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        experiment.predict(setting, load=True)
    else:
        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        experiment.test(setting, test=1)
    torch.cuda.empty_cache()


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
