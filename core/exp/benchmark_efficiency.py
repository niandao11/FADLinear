from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_MODULES = {
    "fadlinear": "models.FADLinear",
    "dlinear": "models.DLinear",
    "patchtst": "models.PatchTST",
    "timemixer": "models.TimeMixer",
    "itransformer": "models.itransformer",
    "wpmixer": "models.WPMixer",
    "filterts": "models.FilterTS",
    "fits": "models.FITS",
}





TARGET_PROJECTABLE_MODELS = {
    "fadlinear",
    "dlinear",
    "patchtst",
    "timemixer",
    "wpmixer",
    "fits",
}


def expected_scope(model_key: str, args: SimpleNamespace) -> Tuple[str, str]:
    if model_key == "timemixer":
        channel_independence = int(getattr(args, "channel_independence", 1))
        if channel_independence == 1:
            return "target_only", "TimeMixer channel_independence=1"
        return "full_input", "TimeMixer channel_independence=0"
    if model_key == "filterts":
        filter_type = str(getattr(args, "filter_type", "all"))
        if filter_type in {"all", "cross_variable"}:
            return "full_input", f"FilterTS filter_type={filter_type} uses cross-variable filtering"
        
        return "full_input", f"FilterTS filter_type={filter_type} is measured conservatively"
    if model_key == "itransformer":
        return "full_input", "iTransformer uses cross-variate attention"
    if model_key in TARGET_PROJECTABLE_MODELS:
        return "target_only", f"{model_key} has a channel-independent target path"
    return "full_input", "unregistered model defaults to full-input measurement"


def validate_scope(model_key: str, declared_scope: str, args: SimpleNamespace) -> str:
    required_scope, reason = expected_scope(model_key, args)
    if declared_scope != required_scope:
        raise ValueError(
            f"Scope mismatch for {model_key}: config declares {declared_scope!r}, "
            f"but instantiated arguments require {required_scope!r} ({reason})."
        )
    return reason


DEFAULT_ARGS: Dict[str, Any] = {
    "task_name": "long_term_forecast",
    "features": "MS",
    "target": "OT",
    "seq_len": 336,
    "label_len": 48,
    "pred_len": 720,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 1,
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 2048,
    "dropout": 0.05,
    "activation": "gelu",
    "embed": "timeF",
    "freq": "h",
    "embed_type": 0,
    "factor": 1,
    "distil": True,
    "output_attention": False,
    "use_norm": True,
    "class_strategy": "cls",
    "individual": False,
    "moving_avg": 25,
    "decomp_kernel": 25,
    "kernel_size": 25,
    "patch_len": 16,
    "stride": 8,
    "padding_patch": "end",
    "fc_dropout": 0.2,
    "head_dropout": 0.0,
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "down_sampling_layers": 3,
    "down_sampling_window": 2,
    "down_sampling_method": "avg",
    "channel_independence": 1,
    "decomp_method": "moving_avg",
    "top_k": 5,
    "use_future_temporal_feature": 0,
    "filter_type": "all",
    "quantile": 0.9,
    "bandwidth": 1,
    "top_K_static_freqs": 10,
    "embedding": "fourier_interpolate",
    "wavelet": "db2",
    "level": 1,
    "tfactor": 5,
    "dfactor": 5,
    "embedding_dropout": None,
    "no_decomposition": False,
    "fits_h_order": 6,
    "fits_base_t": 24,
    "fits_train_mode": 1,
    "batch_size": 1,
    "use_amp": False,
}


MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "fadlinear": {},
    "dlinear": {"individual": False},
    "patchtst": {
        "d_model": 128,
        "d_ff": 256,
        "e_layers": 3,
        "n_heads": 16,
        "dropout": 0.2,
        "fc_dropout": 0.2,
        "head_dropout": 0.0,
        "patch_len": 16,
        "stride": 8,
        "padding_patch": "end",
        "revin": 1,
        "affine": 0,
        "decomposition": 0,
        "individual": False,
    },
    "timemixer": {
        "d_model": 16,
        "d_ff": 32,
        "e_layers": 2,
        "dropout": 0.1,
        "moving_avg": 25,
        "down_sampling_layers": 3,
        "down_sampling_window": 2,
        "down_sampling_method": "avg",
        "channel_independence": 1,
        "decomp_method": "moving_avg",
        "top_k": 5,
    },
    "itransformer": {
        "d_model": 128,
        "d_ff": 128,
        "e_layers": 2,
        "n_heads": 8,
        "dropout": 0.1,
    },
    "wpmixer": {
        "d_model": 128,
        "wavelet": "db2",
        "level": 1,
        "tfactor": 5,
        "dfactor": 5,
        "patch_len": 16,
        "stride": 8,
        "dropout": 0.2,
        "embedding_dropout": None,
        "no_decomposition": False,
    },
    "filterts": {
        "e_layers": 3,
        "factor": 3,
        "d_model": 128,
        "quantile": 0.9,
        "bandwidth": 1,
        "top_K_static_freqs": 10,
        "filter_type": "all",
        "embedding": "fourier_interpolate",
        "use_norm": True,
    },
    "fits": {
        "fits_h_order": 6,
        "fits_base_t": 24,
        "fits_train_mode": 1,
        "individual": False,
    },
}


class ForwardAdapter(nn.Module):

    def __init__(self, model: nn.Module, args: SimpleNamespace, batch_size: int):
        super().__init__()
        self.model = model
        self.seq_len = int(args.seq_len)
        self.label_len = int(getattr(args, "label_len", 0))
        self.pred_len = int(args.pred_len)
        self.channels = int(args.enc_in)
        self.batch_size = int(batch_size)
        mark_dim = int(getattr(args, "mark_dim", 4))
        dec_len = self.label_len + self.pred_len
        self.register_buffer(
            "x_mark_enc", torch.zeros(self.batch_size, self.seq_len, mark_dim), persistent=False
        )
        self.register_buffer(
            "x_dec", torch.zeros(self.batch_size, dec_len, self.channels), persistent=False
        )
        self.register_buffer(
            "x_mark_dec", torch.zeros(self.batch_size, dec_len, mark_dim), persistent=False
        )
        self._forward_params = tuple(inspect.signature(model.forward).parameters)

    def forward(self, x: torch.Tensor) -> Any:
        if x.shape != (self.batch_size, self.seq_len, self.channels):
            raise ValueError(
                f"Expected input {(self.batch_size, self.seq_len, self.channels)}, "
                f"got {tuple(x.shape)}"
            )
        values = {
            "x": x,
            "batch_x": x,
            "x_enc": x,
            "batch_x_mark": self.x_mark_enc,
            "x_mark": self.x_mark_enc,
            "x_mark_enc": self.x_mark_enc,
            "dec_inp": self.x_dec,
            "x_dec": self.x_dec,
            "batch_y_mark": self.x_mark_dec,
            "x_mark_dec": self.x_mark_dec,
        }
        kwargs = {name: values[name] for name in self._forward_params if name in values}
        return self.model(**kwargs)


def normalize_model_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("The benchmark config must be a JSON object.")
    return value


def resolve_checkpoint(raw_path: str, config_dir: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(raw_path)))
    roots = [expanded] if expanded.is_absolute() else [config_dir / expanded, PROJECT_ROOT / expanded]
    file_names = ("checkpoint.pth", "checkpoints.pth", "best.pth")
    candidates: List[Path] = []
    for root in roots:
        if root.is_file():
            candidates.append(root)
            continue
        for name in file_names:
            candidates.extend((root / name, root / "checkpoints" / name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = "\n  ".join(str(path) for path in candidates or roots)
    raise FileNotFoundError(f"Checkpoint not found. Checked:\n  {checked}")


def torch_load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_checkpoint(payload: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    saved_args: Dict[str, Any] = {}
    if isinstance(payload, Mapping):
        raw_args = payload.get("args")
        if raw_args is not None:
            if isinstance(raw_args, Mapping):
                saved_args = dict(raw_args)
            elif hasattr(raw_args, "__dict__"):
                saved_args = dict(vars(raw_args))
        state = payload.get("state_dict") or payload.get("model_state_dict")
        if state is None and payload and all(torch.is_tensor(v) for v in payload.values()):
            state = payload
    else:
        state = None
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint does not contain a recognizable state_dict.")
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            raise ValueError(f"Non-tensor checkpoint entry: {key}")
        clean_key = key[7:] if key.startswith("module.") else key
        cleaned[clean_key] = value
    return cleaned, saved_args


def build_args(
    model_key: str,
    saved_args: Mapping[str, Any],
    common_args: Mapping[str, Any],
    entry_args: Mapping[str, Any],
    batch_size: int,
) -> SimpleNamespace:
    values = copy.deepcopy(DEFAULT_ARGS)
    values.update(copy.deepcopy(MODEL_DEFAULTS.get(model_key, {})))
    values.update(dict(saved_args))
    values.update(dict(common_args))
    values.update(dict(entry_args))
    values["batch_size"] = int(batch_size)
    values["device"] = torch.device("cpu")
    return SimpleNamespace(**values)


def build_model(model_key: str, entry: Mapping[str, Any], args: SimpleNamespace) -> nn.Module:
    module_name = entry.get("module") or MODEL_MODULES.get(model_key)
    if not module_name:
        raise ValueError(f"No module mapping for model {entry.get('model')!r}; set 'module' in JSON.")
    module = importlib.import_module(str(module_name))
    model_class = getattr(module, str(entry.get("class", "Model")))
    return model_class(args).float().eval()


def strict_load(model: nn.Module, state: Mapping[str, torch.Tensor], label: str) -> None:
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"{label} state mismatch: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )


def project_tensor_to_target(
    source: torch.Tensor,
    target_shape: Sequence[int],
    original_channels: int,
    target_index: int,
) -> torch.Tensor:
    if tuple(source.shape) == tuple(target_shape):
        return source
    if source.ndim != len(target_shape):
        raise ValueError(f"rank {source.ndim} cannot become {len(target_shape)}")
    slices: List[slice] = []
    for source_size, target_size in zip(source.shape, target_shape):
        if source_size == target_size:
            slices.append(slice(None))
        elif source_size == original_channels and target_size == 1:
            slices.append(slice(target_index, target_index + 1))
        else:
            raise ValueError(f"dimension {source_size} cannot become {target_size}")
    projected = source[tuple(slices)].clone()
    if tuple(projected.shape) != tuple(target_shape):
        raise ValueError(f"projected shape {tuple(projected.shape)} != {tuple(target_shape)}")
    return projected


def build_target_only_model(
    model_key: str,
    entry: Mapping[str, Any],
    full_args: SimpleNamespace,
    full_state: Mapping[str, torch.Tensor],
    target_index: int,
) -> Tuple[nn.Module, SimpleNamespace, List[str]]:
    if model_key not in TARGET_PROJECTABLE_MODELS:
        raise ValueError(f"{entry.get('model')} is not registered for target-only projection.")
    if bool(getattr(full_args, "individual", False)):
        raise ValueError("target_only currently requires individual=false for strict checkpoint projection.")
    original_channels = int(full_args.enc_in)
    target_args = copy.deepcopy(full_args)
    target_args.enc_in = 1
    target_args.dec_in = 1
    target_args.c_out = 1
    target_args.target_idx = 0
    target_model = build_model(model_key, entry, target_args)
    target_template = target_model.state_dict()
    projected: Dict[str, torch.Tensor] = {}
    projected_keys: List[str] = []
    for key, target_value in target_template.items():
        if key not in full_state:
            raise KeyError(f"Target model key is absent from original checkpoint: {key}")
        try:
            value = project_tensor_to_target(
                full_state[key], target_value.shape, original_channels, target_index
            )
        except ValueError as exc:
            raise ValueError(f"Cannot project state key {key}: {exc}") from exc
        projected[key] = value
        if tuple(value.shape) != tuple(full_state[key].shape):
            projected_keys.append(key)
    strict_load(target_model, projected, "target-only")
    return target_model, target_args, projected_keys


def first_output(value: Any) -> torch.Tensor:
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not torch.is_tensor(value):
        raise TypeError(f"Model output must be a tensor, got {type(value).__name__}.")
    if value.ndim != 3:
        raise ValueError(f"Expected [B,H,C] output, got {tuple(value.shape)}.")
    return value


def select_target_output(value: Any, target_index: int) -> torch.Tensor:
    output = first_output(value)
    index = target_index if output.shape[-1] > target_index else output.shape[-1] - 1
    return output[..., index : index + 1]


def verify_target_equivalence(
    full_model: nn.Module,
    full_args: SimpleNamespace,
    target_model: nn.Module,
    target_args: SimpleNamespace,
    target_index: int,
    atol: float,
    rtol: float,
    trials: int,
    seed: int,
) -> float:
    full_adapter = ForwardAdapter(full_model.eval(), full_args, batch_size=1)
    target_adapter = ForwardAdapter(target_model.eval(), target_args, batch_size=1)
    max_abs_diff = 0.0
    for trial in range(trials):
        generator = torch.Generator(device="cpu").manual_seed(seed + trial)
        x_full = torch.randn(
            1, int(full_args.seq_len), int(full_args.enc_in), generator=generator
        )
        x_target = x_full[..., target_index : target_index + 1]
        with torch.inference_mode():
            full_output = select_target_output(full_adapter(x_full), target_index)
            target_output = select_target_output(target_adapter(x_target), 0)
        trial_diff = float((full_output - target_output).abs().max().item())
        max_abs_diff = max(max_abs_diff, trial_diff)
        if not torch.allclose(full_output, target_output, atol=atol, rtol=rtol):
            raise RuntimeError(
                "Target-path equivalence failed: "
                f"trial={trial + 1}/{trials}, max_abs_diff={trial_diff:.8g}, "
                f"atol={atol}, rtol={rtol}"
            )
    return max_abs_diff


def verify_non_target_invariance(
    full_model: nn.Module,
    full_args: SimpleNamespace,
    target_index: int,
    atol: float,
    rtol: float,
    trials: int,
    seed: int,
) -> float:
    channels = int(full_args.enc_in)
    if channels == 1:
        return 0.0
    adapter = ForwardAdapter(full_model.eval(), full_args, batch_size=1)
    non_target = [index for index in range(channels) if index != target_index]
    max_abs_diff = 0.0
    for trial in range(trials):
        generator = torch.Generator(device="cpu").manual_seed(seed + trial)
        baseline = torch.randn(1, int(full_args.seq_len), channels, generator=generator)
        perturbed = baseline.clone()
        perturbation = torch.randn(
            1, int(full_args.seq_len), len(non_target), generator=generator
        )
        perturbed[..., non_target] = perturbation * 3.0 + 1.0
        with torch.inference_mode():
            baseline_output = select_target_output(adapter(baseline), target_index)
            perturbed_output = select_target_output(adapter(perturbed), target_index)
        trial_diff = float((baseline_output - perturbed_output).abs().max().item())
        max_abs_diff = max(max_abs_diff, trial_diff)
        if not torch.allclose(baseline_output, perturbed_output, atol=atol, rtol=rtol):
            raise RuntimeError(
                "Non-target invariance failed: "
                f"trial={trial + 1}/{trials}, max_abs_diff={trial_diff:.8g}, "
                f"atol={atol}, rtol={rtol}"
            )
    return max_abs_diff


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)


class FlopCoverageError(RuntimeError):
    def __init__(self, message: str, audit: Sequence[Mapping[str, Any]], coverage: float):
        super().__init__(message)
        self.audit = list(audit)
        self.coverage = float(coverage)


def _tensor_leaves(value: Any) -> List[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, Mapping):
        leaves: List[torch.Tensor] = []
        for item in value.values():
            leaves.extend(_tensor_leaves(item))
        return leaves
    if isinstance(value, (tuple, list)):
        leaves = []
        for item in value:
            leaves.extend(_tensor_leaves(item))
        return leaves
    return []


def _output_numel(value: Any) -> int:
    return sum(int(tensor.numel()) for tensor in _tensor_leaves(value))


def _is_complex_value(value: Any) -> bool:
    return any(tensor.is_complex() for tensor in _tensor_leaves(value))


def _prod(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


ZERO_FLOP_OPS = {
    "aten.alias", "aten.arange", "aten.as_strided", "aten.cat", "aten.clone",
    "aten.complex", "aten.conj", "aten.constant_pad_nd", "aten.contiguous",
    "aten.copy_", "aten.detach", "aten.empty", "aten.empty_like", "aten.empty_strided",
    "aten.eq", "aten.expand", "aten.fill_", "aten.flip", "aten.full", "aten.full_like",
    "aten.ge", "aten.gt", "aten.index", "aten.index_put_", "aten.isfinite", "aten.le",
    "aten.flatten", "aten.imag", "aten.lift_fresh", "aten.logical_and", "aten.logical_not",
    "aten.logical_or", "aten.lt",
    "aten.masked_fill", "aten.max", "aten.maximum", "aten.min", "aten.minimum", "aten.ne",
    "aten.new_empty", "aten.new_zeros", "aten.ones", "aten.ones_like", "aten.permute",
    "aten.pad", "aten.real", "aten.reflection_pad1d", "aten.reflection_pad2d", "aten.repeat",
    "aten.replication_pad1d", "aten.reshape", "aten.resolve_conj", "aten.roll", "aten.select",
    "aten.slice", "aten.sort", "aten.split", "aten.split_with_sizes", "aten.squeeze",
    "aten.stack", "aten.t", "aten.to", "aten.topk", "aten.transpose", "aten.unbind",
    "aten.unfold", "aten.unsqueeze", "aten.view", "aten.view_as_complex", "aten.view_as_real",
    "aten.where", "aten.zeros", "aten.zeros_like", "aten._to_copy", "aten._unsafe_view",
}


def _conv_flops(
    name: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    output: Any,
) -> int:
    source, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    transposed = "conv_transpose" in name
    if name in {"aten.convolution", "aten._convolution"}:
        transposed = bool(args[6]) if len(args) > 6 else bool(kwargs.get("transposed", False))

    if transposed:
        
        
        
        spatial = _prod(source.shape[2:])
        flops = int(source.shape[0]) * spatial * int(weight.numel()) * 2
    else:
        
        
        kernel_work = int(weight[0].numel())
        flops = _output_numel(output) * kernel_work * 2
    if bias is not None:
        flops += _output_numel(output)
    return int(flops)


def _expand_einsum_term(term: str, rank: int, ellipsis_labels: Sequence[str]) -> List[str]:
    explicit_count = len(term.replace("...", ""))
    ellipsis_rank = rank - explicit_count
    if ellipsis_rank < 0:
        raise ValueError(f"Invalid einsum term {term!r} for rank {rank}.")
    if "..." not in term:
        if ellipsis_rank != 0:
            raise ValueError(f"Einsum term {term!r} does not describe rank {rank}.")
        return list(term)
    selected = list(ellipsis_labels[len(ellipsis_labels) - ellipsis_rank :])
    prefix, suffix = term.split("...", 1)
    return list(prefix) + selected + list(suffix)


def _einsum_flops(args: Sequence[Any], output: Any, complex_op: bool) -> int:
    equation = str(args[0]).replace(" ", "")
    operands_raw = args[1] if len(args) > 1 else ()
    operands = list(operands_raw) if isinstance(operands_raw, (tuple, list)) else list(args[1:])
    if not operands:
        raise ValueError(f"Einsum {equation!r} has no operands.")

    if "->" in equation:
        input_equation, output_equation = equation.split("->", 1)
    else:
        input_equation, output_equation = equation, None
    input_terms = input_equation.split(",")
    if len(input_terms) != len(operands):
        raise ValueError(
            f"Einsum {equation!r} has {len(input_terms)} terms for {len(operands)} operands."
        )

    max_ellipsis_rank = max(
        int(operand.ndim) - len(term.replace("...", ""))
        for term, operand in zip(input_terms, operands)
    )
    ellipsis_labels = [f"@ELLIPSIS_{index}" for index in range(max_ellipsis_rank)]
    expanded_terms = [
        _expand_einsum_term(term, int(operand.ndim), ellipsis_labels)
        for term, operand in zip(input_terms, operands)
    ]

    label_sizes: Dict[str, int] = {}
    label_occurrences: Dict[str, int] = {}
    for labels, operand in zip(expanded_terms, operands):
        if len(labels) != int(operand.ndim):
            raise ValueError(f"Expanded einsum labels do not match shape {tuple(operand.shape)}.")
        for label, size in zip(labels, operand.shape):
            size_int = int(size)
            previous = label_sizes.get(label, 1)
            if previous != size_int and previous != 1 and size_int != 1:
                raise ValueError(f"Incompatible einsum dimension for label {label!r}.")
            label_sizes[label] = max(previous, size_int)
            label_occurrences[label] = label_occurrences.get(label, 0) + 1

    if output_equation is None:
        output_labels = list(ellipsis_labels)
        output_labels.extend(
            sorted(label for label, count in label_occurrences.items() if count == 1 and label[0] != "@")
        )
    elif "..." in output_equation:
        prefix, suffix = output_equation.split("...", 1)
        output_labels = list(prefix) + list(ellipsis_labels) + list(suffix)
    else:
        output_labels = list(output_equation)

    reduction_labels = set(label_sizes).difference(output_labels)
    reduction_size = _prod(label_sizes[label] for label in reduction_labels)
    output_elements = _output_numel(output)
    multiplication_count = max(0, len(operands) - 1) * reduction_size
    addition_count = max(0, reduction_size - 1)
    if complex_op:
        
        
        return output_elements * (multiplication_count * 6 + addition_count * 2)
    return output_elements * (multiplication_count + addition_count)


def _fft_1d_flops(
    name: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    output: Any,
) -> int:
    source = args[0]
    output_tensor = _tensor_leaves(output)[0]
    dim_arg = args[2] if len(args) > 2 and args[2] is not None else kwargs.get("dim", -1)
    dim = int(dim_arg)
    if dim < 0:
        dim += int(source.ndim)
    n_arg = args[1] if len(args) > 1 else kwargs.get("n")

    if name == "aten.fft_irfft":
        transformed = int(n_arg) if n_arg is not None else int(output_tensor.shape[dim])
        transforms = int(output_tensor.numel()) // transformed
        coefficient = 2.5
    else:
        transformed = int(n_arg) if n_arg is not None else int(source.shape[dim])
        transforms = int(source.numel()) // int(source.shape[dim])
        coefficient = 2.5 if name == "aten.fft_rfft" else 5.0
    flops = coefficient * transforms * transformed * math.log2(max(2, transformed))
    return int(math.ceil(flops))


def _complete_flop_formula(
    name: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    output: Any,
) -> Tuple[Optional[int], str]:
    tensors = _tensor_leaves((args, kwargs))
    out_numel = _output_numel(output)
    complex_op = _is_complex_value((args, output))

    if (
        name in ZERO_FLOP_OPS
        or name.startswith("aten.sym_")
        or name.startswith("aten.conj")
        or name.startswith("aten._conj")
    ):
        return 0, "data movement/indexing/comparison"
    if name in {"aten.mm", "aten.bmm", "aten.matmul"}:
        left, right = args[0], args[1]
        inner = int(left.shape[-1])
        return out_numel * inner * (8 if complex_op else 2), "complex matmul" if complex_op else "matmul"
    if name in {"aten.addmm", "aten.baddbmm"}:
        left, right = args[1], args[2]
        inner = int(left.shape[-1])
        mac_cost = 8 if complex_op else 2
        bias_cost = 2 if complex_op else 1
        return out_numel * (inner * mac_cost + bias_cost), "matmul plus bias"
    if name == "aten.linear":
        source, weight = args[0], args[1]
        inner = int(weight.shape[-1])
        has_bias = len(args) > 2 and args[2] is not None
        bias_cost = (2 if complex_op else 1) if has_bias else 0
        return out_numel * (inner * (8 if complex_op else 2) + bias_cost), "linear"
    if name in {
        "aten.convolution", "aten._convolution", "aten.cudnn_convolution",
        "aten._slow_conv2d_forward", "aten.conv1d", "aten.conv2d", "aten.conv3d",
        "aten.conv_transpose1d", "aten.conv_transpose2d", "aten.conv_transpose3d",
    }:
        return _conv_flops(name, args, kwargs, output), "direct convolution"
    if name == "aten.einsum":
        return _einsum_flops(args, output, complex_op), "einsum contraction"
    if name in {"aten._fft_r2c", "aten._fft_c2r", "aten._fft_c2c"}:
        source = args[0]
        dims = tuple(int(dim) for dim in args[1])
        transformed = _prod(source.shape[dim] for dim in dims)
        if name == "aten._fft_c2r":
            out_tensor = _tensor_leaves(output)[0]
            transformed = _prod(out_tensor.shape[dim] for dim in dims)
            transforms = int(out_tensor.numel()) // transformed
            coefficient = 2.5
        else:
            transforms = int(source.numel()) // transformed
            coefficient = 2.5 if name == "aten._fft_r2c" else 5.0
        flops = coefficient * transforms * transformed * math.log2(max(2, transformed))
        return int(math.ceil(flops)), "FFT real-equivalent convention"
    if name in {"aten.fft_rfft", "aten.fft_irfft", "aten.fft_fft", "aten.fft_ifft"}:
        return _fft_1d_flops(name, args, kwargs, output), "FFT real-equivalent convention"
    if "layer_norm" in name:
        source = args[0]
        normalized_shape = args[1]
        normalized = _prod(normalized_shape)
        groups = int(source.numel()) // normalized
        affine = len(args) > 2 and args[2] is not None
        return groups * normalized * (8 if affine else 6), "layer normalization"
    if "batch_norm" in name:
        source = args[0]
        training = bool(args[5]) if len(args) > 5 and isinstance(args[5], bool) else False
        return int(source.numel()) * (8 if training else 4), "batch normalization"
    if name in {"aten.mean", "aten.sum"}:
        source = args[0]
        if name == "aten.mean":
            return int(source.numel()), "reduction mean"
        return max(0, int(source.numel()) - out_numel), "reduction sum"
    if name in {"aten.var", "aten.var_mean", "aten.std", "aten.std_mean"}:
        source = args[0]
        per_element = 4 + (1 if "std" in name else 0)
        return int(source.numel()) * per_element, "variance/standard deviation"
    if name in {"aten._softmax", "aten.softmax"}:
        return out_numel * 4, "softmax"
    if name in {"aten._log_softmax", "aten.log_softmax"}:
        return out_numel * 5, "log-softmax"
    if name in {"aten.avg_pool1d", "aten.avg_pool2d", "aten.avg_pool3d"}:
        kernel = args[1]
        kernel_work = _prod(kernel if isinstance(kernel, (tuple, list)) else (kernel,))
        return out_numel * kernel_work, "average pooling"
    if name.startswith("aten.adaptive_avg_pool"):
        source = args[0]
        return int(source.numel()), "adaptive average pooling"
    if name in {"aten.upsample_linear1d", "aten.upsample_bilinear2d", "aten.upsample_trilinear3d"}:
        dimensions = {"aten.upsample_linear1d": 1, "aten.upsample_bilinear2d": 2, "aten.upsample_trilinear3d": 3}[name]
        return out_numel * (3 ** dimensions), "linear interpolation"
    if name == "aten.quantile":
        
        
        return out_numel * 3, "quantile interpolation"
    if name in {"aten.add", "aten.sub", "aten.rsub"}:
        return out_numel * (2 if complex_op else 1), "elementwise add/sub"
    if name == "aten.mul":
        return out_numel * (6 if complex_op else 1), "elementwise multiply"
    if name.startswith("aten.div") or name.startswith("aten.true_divide"):
        return out_numel * (11 if complex_op else 1), "elementwise divide"
    if name in {"aten.neg", "aten.square", "aten.pow", "aten.reciprocal", "aten.sqrt", "aten.exp", "aten.log"}:
        return out_numel, "elementwise arithmetic/transcendental"
    if name == "aten.rsqrt":
        return out_numel * 2, "reciprocal square root"
    if name == "aten.abs":
        return out_numel * (4 if tensors and tensors[0].is_complex() else 0), "absolute magnitude"
    if name in {"aten.sigmoid"}:
        return out_numel * 3, "sigmoid"
    if name in {"aten.tanh"}:
        return out_numel, "tanh"
    if name in {"aten.gelu"}:
        return out_numel * 8, "GELU"
    if name in {"aten.silu"}:
        return out_numel * 4, "SiLU"
    if name in {"aten.relu", "aten.clamp", "aten.clamp_min", "aten.clamp_max"}:
        return 0, "comparison/selection activation"
    if name in {"aten.lerp"}:
        return out_numel * (8 if complex_op else 3), "linear interpolation blend"
    if name in {"aten.addcmul"}:
        return out_numel * (8 if complex_op else 2), "fused multiply-add"
    if name in {"aten.native_dropout", "aten.dropout"}:
        training = bool(args[2]) if len(args) > 2 else False
        return out_numel if training else 0, "dropout scaling"
    return None, "unsupported"


class CompleteFlopCounterMode:

    def __init__(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        owner = self

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                output = func(*args, **kwargs)
                packet = func._overloadpacket
                name = str(packet)
                flops, rule = _complete_flop_formula(name, args, kwargs, output)
                owner._record(name, flops, rule, args, output)
                return output

        self.mode = _Mode()
        self.records: Dict[str, Dict[str, Any]] = {}

    def _record(self, name: str, flops: Optional[int], rule: str, args: Any, output: Any) -> None:
        record = self.records.setdefault(
            name,
            {"operator": name, "calls": 0, "flops": 0, "status": "covered", "rule": rule, "dtypes": set()},
        )
        record["calls"] += 1
        record["rule"] = rule
        record["dtypes"].update(str(tensor.dtype) for tensor in _tensor_leaves((args, output)))
        if flops is None:
            record["status"] = "unsupported"
        else:
            record["flops"] += int(flops)

    def __enter__(self):
        self.mode.__enter__()
        return self

    def __exit__(self, *args):
        return self.mode.__exit__(*args)

    def audit(self) -> List[Dict[str, Any]]:
        rows = []
        for record in sorted(self.records.values(), key=lambda item: item["operator"]):
            row = dict(record)
            row["dtypes"] = ";".join(sorted(row["dtypes"]))
            rows.append(row)
        return rows

    def coverage(self) -> float:
        total = sum(int(record["calls"]) for record in self.records.values())
        covered = sum(
            int(record["calls"]) for record in self.records.values() if record["status"] == "covered"
        )
        return 100.0 if total == 0 else 100.0 * covered / total

    def total_flops(self) -> int:
        return sum(int(record["flops"]) for record in self.records.values())


def measure_flops_complete(
    adapter: nn.Module, args: SimpleNamespace
) -> Tuple[Optional[float], float, int, List[Dict[str, Any]], float]:
    adapter = adapter.cpu().eval()
    sample = torch.randn(1, int(args.seq_len), int(args.enc_in), dtype=torch.float32)
    counter = CompleteFlopCounterMode()
    with torch.inference_mode(), counter:
        adapter(sample)
    audit = counter.audit()
    coverage = counter.coverage()
    unsupported = [row["operator"] for row in audit if row["status"] == "unsupported"]
    if unsupported:
        raise FlopCoverageError(
            "Complete FLOP coverage failed; unsupported operators: " + ", ".join(unsupported),
            audit,
            coverage,
        )
    flops = float(counter.total_flops())
    if flops <= 0:
        raise FlopCoverageError("Complete FLOP counter returned zero FLOPs.", audit, coverage)
    _, trainable = count_parameters(adapter.model)
    return None, flops, trainable, audit, coverage


def measure_flops_ptflops(adapter: nn.Module, args: SimpleNamespace) -> Tuple[float, float, int]:
    try:
        from ptflops import get_model_complexity_info
    except ImportError as exc:
        raise RuntimeError("ptflops is required for --flops-backend ptflops.") from exc
    adapter = adapter.cpu().eval()
    macs, reported_params = get_model_complexity_info(
        adapter,
        (int(args.seq_len), int(args.enc_in)),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    if macs is None or not math.isfinite(float(macs)) or float(macs) <= 0:
        raise RuntimeError(f"ptflops returned invalid MACs: {macs}")
    return float(macs), float(macs) * 2.0, int(reported_params)


def measure_flops_profiler(adapter: nn.Module, args: SimpleNamespace) -> Tuple[float, float, int]:
    adapter = adapter.cpu().eval()
    sample = torch.randn(1, int(args.seq_len), int(args.enc_in))
    with torch.inference_mode(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True
    ) as profile:
        adapter(sample)
    flops = float(sum(float(event.flops or 0) for event in profile.key_averages()))
    if flops <= 0:
        raise RuntimeError("torch.profiler returned zero FLOPs.")
    _, trainable = count_parameters(adapter.model)
    return flops / 2.0, flops, trainable


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_latency(
    adapter: nn.Module,
    args: SimpleNamespace,
    device: torch.device,
    batch_size: int,
    warmup: int,
    runs: int,
    trials: int,
) -> Tuple[float, float, List[float]]:
    adapter = adapter.to(device).eval()
    sample = torch.randn(batch_size, int(args.seq_len), int(args.enc_in), device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            adapter(sample)
        synchronize(device)
        timings: List[float] = []
        for _ in range(trials):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(runs):
                    adapter(sample)
                end.record()
                synchronize(device)
                elapsed_ms = float(start.elapsed_time(end)) / runs
            else:
                started = time.perf_counter()
                for _ in range(runs):
                    adapter(sample)
                elapsed_ms = (time.perf_counter() - started) * 1000.0 / runs
            timings.append(elapsed_ms)
    median_ms = float(statistics.median(timings))
    iqr_ms = percentile(timings, 0.75) - percentile(timings, 0.25)
    return median_ms, float(iqr_ms), timings


def measure_peak_memory(
    adapter: nn.Module,
    args: SimpleNamespace,
    device: torch.device,
    batch_size: int,
    runs: int,
) -> Tuple[Optional[float], Optional[float]]:
    if device.type != "cuda":
        return None, None
    adapter = adapter.to(device).eval()
    sample = torch.randn(batch_size, int(args.seq_len), int(args.enc_in), device=device)
    gc.collect()
    torch.cuda.empty_cache()
    synchronize(device)
    baseline = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(runs):
            adapter(sample)
    synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    return peak / (1024.0 ** 2), max(0, peak - baseline) / (1024.0 ** 2)


def runtime_metadata(device: torch.device) -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def run_entry(
    entry: Mapping[str, Any],
    common_args: Mapping[str, Any],
    config_dir: Path,
    cli: argparse.Namespace,
) -> Dict[str, Any]:
    display_name = str(entry.get("name") or entry.get("model"))
    model_key = normalize_model_name(str(entry.get("model", display_name)))
    scope = str(entry.get("scope", "full_input"))
    if scope not in {"target_only", "full_input"}:
        raise ValueError(f"Invalid scope for {display_name}: {scope}")
    checkpoint_path = resolve_checkpoint(str(entry["checkpoint"]), config_dir)
    payload = torch_load_checkpoint(checkpoint_path)
    state, saved_args = extract_checkpoint(payload)
    full_args = build_args(
        model_key,
        saved_args,
        common_args,
        entry.get("args", {}),
        cli.batch_size,
    )
    scope_reason = validate_scope(model_key, scope, full_args)
    original_channels = int(full_args.enc_in)
    target_index = int(entry.get("target_index", original_channels - 1))
    if not 0 <= target_index < original_channels:
        raise ValueError(f"target_index={target_index} is invalid for enc_in={original_channels}.")

    full_model = build_model(model_key, entry, full_args)
    strict_load(full_model, state, "original")
    full_total_params, full_trainable_params = count_parameters(full_model)
    projected_keys: List[str] = []
    equivalence_diff: Optional[float] = None
    non_target_invariance_diff: Optional[float] = None
    if scope == "target_only":
        non_target_invariance_diff = verify_non_target_invariance(
            full_model,
            full_args,
            target_index,
            cli.equivalence_atol,
            cli.equivalence_rtol,
            cli.verification_trials,
            cli.seed + 1000,
        )
        measured_model, measured_args, projected_keys = build_target_only_model(
            model_key, entry, full_args, state, target_index
        )
        equivalence_diff = verify_target_equivalence(
            full_model,
            full_args,
            measured_model,
            measured_args,
            target_index,
            cli.equivalence_atol,
            cli.equivalence_rtol,
            cli.verification_trials,
            cli.seed + 2000,
        )
    else:
        measured_model, measured_args = full_model, full_args

    total_params, trainable_params = count_parameters(measured_model)
    flop_adapter = ForwardAdapter(measured_model, measured_args, batch_size=1)
    flop_operator_audit: List[Dict[str, Any]] = []
    flop_coverage_percent: Optional[float] = None
    if cli.flops_backend == "ptflops":
        macs, flops, profiler_params = measure_flops_ptflops(flop_adapter, measured_args)
    elif cli.flops_backend == "profiler":
        macs, flops, profiler_params = measure_flops_profiler(flop_adapter, measured_args)
    else:
        macs, flops, profiler_params, flop_operator_audit, flop_coverage_percent = (
            measure_flops_complete(flop_adapter, measured_args)
        )
    if profiler_params != trainable_params:
        print(
            f"WARNING: {display_name} parameter disagreement: "
            f"direct={trainable_params}, {cli.flops_backend}={profiler_params}",
            file=sys.stderr,
        )

    del flop_adapter
    if measured_model is not full_model:
        del full_model
    gc.collect()
    if cli.device.type == "cuda":
        torch.cuda.empty_cache()

    if cli.flops_audit_only:
        latency_adapter = None
        latency_median = None
        latency_iqr = None
        latency_trials = []
        peak_memory = None
        incremental_memory = None
    else:
        measured_args.device = cli.device
        latency_adapter = ForwardAdapter(measured_model, measured_args, batch_size=cli.batch_size)
        latency_median, latency_iqr, latency_trials = measure_latency(
            latency_adapter,
            measured_args,
            cli.device,
            cli.batch_size,
            cli.warmup,
            cli.runs,
            cli.trials,
        )
        peak_memory, incremental_memory = measure_peak_memory(
            latency_adapter,
            measured_args,
            cli.device,
            cli.batch_size,
            cli.memory_runs,
        )

    result = {
        "status": "PASS",
        "name": display_name,
        "model": str(entry.get("model")),
        "scope": scope,
        "scope_reason": scope_reason,
        "original_channels": original_channels,
        "measured_channels": int(measured_args.enc_in),
        "target_index": target_index if scope == "target_only" else None,
        "seq_len": int(measured_args.seq_len),
        "pred_len": int(measured_args.pred_len),
        "batch_size": cli.batch_size,
        "benchmark_mode": "flops_audit_only" if cli.flops_audit_only else "full_efficiency",
        "full_total_params": full_total_params,
        "full_trainable_params": full_trainable_params,
        "effective_total_params": total_params,
        "effective_trainable_params": trainable_params,
        
        
        "total_params": total_params,
        "trainable_params": trainable_params,
        "macs": macs,
        "flops": flops,
        "flops_backend": cli.flops_backend,
        "flop_coverage_percent": flop_coverage_percent,
        "flop_operator_audit": flop_operator_audit,
        "latency_median_ms": latency_median,
        "latency_iqr_ms": latency_iqr,
        "latency_trials_ms": latency_trials,
        "peak_memory_mb": peak_memory,
        "incremental_peak_memory_mb": incremental_memory,
        "target_equivalence_max_abs_diff": equivalence_diff,
        "non_target_invariance_max_abs_diff": non_target_invariance_diff,
        "projected_state_keys": projected_keys,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    if latency_adapter is not None:
        del latency_adapter
    del measured_model
    gc.collect()
    if cli.device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_outputs(
    output_dir: Path,
    config_path: Path,
    config: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    cli: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "efficiency.json"
    csv_path = output_dir / "efficiency.csv"
    audit_path = output_dir / "flops_operator_audit.csv"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "runtime": dict(metadata),
        "protocol": {
            "batch_size": cli.batch_size,
            "benchmark_mode": "flops_audit_only" if cli.flops_audit_only else "full_efficiency",
            "warmup": cli.warmup,
            "runs_per_trial": cli.runs,
            "trials": cli.trials,
            "memory_runs": cli.memory_runs,
            "flops_backend": cli.flops_backend,
            "flops_definition": (
                "real-equivalent complete forward arithmetic"
                if cli.flops_backend == "complete"
                else "2 FLOPs per MAC"
            ),
            "complex_mac_definition": "8 real FLOPs per complex multiply-accumulate",
            "fft_definition": (
                "2.5*N*log2(N) for real FFT/IFFT; 5*N*log2(N) for complex FFT"
            ),
            "non_arithmetic_policy": "data movement, indexing, and comparisons count as 0 FLOPs",
            "seed": cli.seed,
            "verification_trials": cli.verification_trials,
            "target_only_policy": (
                "configuration-aware scope validation, non-target perturbation invariance, "
                "strict checkpoint projection, and numerical target-output equivalence"
            ),
        },
        "source_config": config,
        "results": list(results),
    }
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    columns = [
        "status",
        "name",
        "model",
        "scope",
        "scope_reason",
        "original_channels",
        "measured_channels",
        "target_index",
        "seq_len",
        "pred_len",
        "batch_size",
        "benchmark_mode",
        "full_trainable_params",
        "effective_trainable_params",
        "trainable_params",
        "flops",
        "flop_coverage_percent",
        "latency_median_ms",
        "latency_iqr_ms",
        "peak_memory_mb",
        "target_equivalence_max_abs_diff",
        "non_target_invariance_max_abs_diff",
        "checkpoint",
        "checkpoint_sha256",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    audit_columns = ["name", "model", "operator", "calls", "flops", "status", "rule", "dtypes"]
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_columns, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            for audit_row in result.get("flop_operator_audit", []) or []:
                writer.writerow(
                    {
                        "name": result.get("name"),
                        "model": result.get("model"),
                        **dict(audit_row),
                    }
                )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="JSON benchmark profile")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200, help="timed forwards per trial")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--memory-runs", type=int, default=10)
    parser.add_argument(
        "--flops-backend",
        choices=("ptflops", "profiler", "complete"),
        default="complete",
    )
    parser.add_argument(
        "--flops-audit-only",
        action="store_true",
        help="run one strict FLOP-counting forward per model and skip latency/memory timing",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--equivalence-atol", type=float, default=1e-5)
    parser.add_argument("--equivalence-rtol", type=float, default=1e-5)
    parser.add_argument(
        "--verification-trials",
        type=int,
        default=5,
        help="fixed-seed trials for target projection and non-target invariance checks",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parsed = parser.parse_args(argv)
    if min(
        parsed.batch_size,
        parsed.warmup,
        parsed.runs,
        parsed.trials,
        parsed.memory_runs,
        parsed.verification_trials,
    ) <= 0:
        parser.error("batch size and benchmark repetition counts must all be positive")
    parsed.device = torch.device(parsed.device)
    if parsed.device.type == "cuda" and not torch.cuda.is_available():
        parser.error(f"CUDA device requested but CUDA is unavailable: {parsed.device}")
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    cli = parse_args(argv)
    if cli.device.type == "cuda":
        torch.cuda.set_device(cli.device)
    random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)

    config_path = cli.config.resolve()
    config = load_json(config_path)
    common_args = config.get("common_args", {})
    entries = [entry for entry in config.get("models", []) if entry.get("enabled", True)]
    if not entries:
        raise ValueError("No enabled model entries in benchmark config.")

    results: List[Dict[str, Any]] = []
    any_failed = False
    for index, entry in enumerate(entries, start=1):
        name = entry.get("name") or entry.get("model") or f"entry-{index}"
        print(f"[{index}/{len(entries)}] Benchmarking {name}", flush=True)
        try:
            result = run_entry(entry, common_args, config_path.parent, cli)
            if cli.flops_audit_only:
                print(
                    f"  PASS params={result['trainable_params']} flops={result['flops']:.0f} "
                    f"coverage={result['flop_coverage_percent']:.2f}%",
                    flush=True,
                )
            else:
                print(
                    f"  PASS params={result['trainable_params']} flops={result['flops']:.0f} "
                    f"latency={result['latency_median_ms']:.4f} ms "
                    f"memory={result['peak_memory_mb']} MB",
                    flush=True,
                )
        except Exception as exc:
            any_failed = True
            result = {
                "status": "FAIL",
                "name": str(name),
                "model": str(entry.get("model", "")),
                "scope": str(entry.get("scope", "")),
                "checkpoint": str(entry.get("checkpoint", "")),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            if isinstance(exc, FlopCoverageError):
                result["flop_operator_audit"] = exc.audit
                result["flop_coverage_percent"] = exc.coverage
            print(f"  FAIL {result['error']}", file=sys.stderr, flush=True)
            if cli.fail_fast:
                results.append(result)
                break
        results.append(result)

    metadata = runtime_metadata(cli.device)
    write_outputs(cli.output_dir, config_path, config, results, metadata, cli)
    print(f"Results: {cli.output_dir / 'efficiency.csv'}")
    print(f"Manifest: {cli.output_dir / 'efficiency.json'}")
    print(f"FLOP audit: {cli.output_dir / 'flops_operator_audit.csv'}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
