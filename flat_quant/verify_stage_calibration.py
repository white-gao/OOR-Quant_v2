"""Matched single-block calibration pilot; all other blocks stay frozen.

This changes training coverage/reconstruction only. Inference uses the existing
FlatQuant finalization, with one shared set of weights and transforms. The
optional weighted-MSE callback has no effect on the production CLI/defaults.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .evaluate_flatquant_alignment_diagnostics import _checkpoint_config
from .evaluate_lfq_boundary_diagnostics import compute_slot_diagnostics
from .flatquant import restore_flatquant_core_layers_from_checkpoints
from .flatquant.runtime import (
    _TrainableFlatQuantBlock, _finalize_flat_block, _flat_wrappers,
    _load_flatquant_checkpoint, _restore_flatquant_train_state, _train_flat_block,
)
from .omniquant.runtime import _first_tensor, _tree_cpu
from .run_m1_onerec_ad import (
    DEFAULT_DATA_DIR, DEFAULT_MODEL_PATH, SID_SLOT_NAMES,
    build_lfq_sid_slot_batches, default_calib_split, load_task_data,
    set_seed, sid_slot_token_ids,
)


ARMS = ("prefill_mse", "complete_mse", "balanced_mse")
ROLES = {"history": slice(None, -3), "a": slice(-3, -2),
         "b": slice(-2, -1), "c": slice(-1, None)}


def balanced_mse(prediction, target):
    """Equal mean MSE for history/A/B/C; no energy normalization.

Only unpadded single-example prompt+A+B batches are supported. The final three
positions PREDICT A/B/C; they are not the supplied A/B/C token identities.
"""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Expected matching [1, tokens, channels] tensors.")
    if prediction.shape[0] != 1 or prediction.shape[1] < 4:
        raise ValueError("Stage MSE needs one unpadded example with nonempty history.")
    return torch.stack([
        F.mse_loss(prediction[:, selected].float(), target[:, selected].float())
        for selected in ROLES.values()
    ]).mean()


def prefill_batch(complete):
    if set(complete) != {"input_ids", "attention_mask"}:
        raise ValueError("Unexpected tokenizer fields; refusing implicit position slicing.")
    if complete["input_ids"].shape[1] < 4 or not complete["attention_mask"].all():
        raise ValueError("Expected an unpadded prompt+A+B batch.")
    return {key: value[:, :-2].clone() for key, value in complete.items()}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


class _CapturedBlockInput(Exception):
    pass


@torch.no_grad()
def capture_inputs(model, layer_idx, batches, label):
    """Stop at the selected block; cache independent FP/quant prefix streams."""
    captured = []

    def hook(_module, args, kwargs):
        captured.append(_tree_cpu((args, kwargs)))
        raise _CapturedBlockInput

    handle = model.model.layers[layer_idx].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for batch in tqdm(batches, desc=label):
            try:
                model(**{k: v.to(model.device) for k, v in batch.items()}, use_cache=False)
            except _CapturedBlockInput:
                pass
    finally:
        handle.remove()
    if len(captured) != len(batches):
        raise RuntimeError("Incomplete block input capture.")
    return captured


@torch.no_grad()
def evaluate(teacher, student, batches, sample_ids, slot_ids, layer_indices, label):
    """Same complete GT prefix in both models; accumulated residual-stream MSE."""
    references, layer_metrics, handles = {}, {}, []
    for layer_idx in layer_indices:
        def capture(_module, _args, output, index=layer_idx):
            references[index] = _first_tensor(output).detach()

        def compare(_module, _args, output, index=layer_idx):
            target = references.pop(index)
            actual = _first_tensor(output)
            metrics = {}
            for role, selected in {"all": slice(None), **ROLES}.items():
                t, q = target[:, selected].float(), actual[:, selected].float()
                squared_error, target_energy = float((q - t).square().sum()), float(t.square().sum())
                metrics[role] = {"squared_error": squared_error, "target_energy": target_energy,
                                 "numel": t.numel(), "mse": squared_error / t.numel(),
                                 "relative_mse": squared_error / max(target_energy, 1e-20)}
            layer_metrics[str(index)] = metrics

        handles.append(teacher.model.layers[layer_idx].register_forward_hook(capture))
        handles.append(student.model.layers[layer_idx].register_forward_hook(compare))
    rows = []
    try:
        for sample_id, batch in tqdm(list(zip(sample_ids, batches)), desc=label):
            batch = {k: v.to(student.device) for k, v in batch.items()}
            t = teacher(**batch, use_cache=False, logits_to_keep=3).logits[0].float()
            q = student(**batch, use_cache=False, logits_to_keep=3).logits[0].float()
            slots = {}
            for i, slot in enumerate(SID_SLOT_NAMES):
                metrics = compute_slot_diagnostics(
                    t[i].index_select(-1, slot_ids[slot]), q[i].index_select(-1, slot_ids[slot]),
                    topk=32, negative_count=32, tie_threshold=0.01, gap_scale=1.0,
                )
                slots[slot] = metrics
            rows.append({"sample_id": sample_id, "slots": slots, "layers": dict(layer_metrics)})
    finally:
        for handle in handles:
            handle.remove()
    return rows


def summarize(rows):
    result = {"samples": len(rows), "slots": {}, "layers": {}}
    for slot in SID_SLOT_NAMES:
        result["slots"][slot] = {
            key: float(np.mean([row["slots"][slot][key] for row in rows]))
            for key in rows[0]["slots"][slot]
        }
    for layer in rows[0]["layers"]:
        result["layers"][layer] = {}
        for role in rows[0]["layers"][layer]:
            values = [row["layers"][layer][role] for row in rows]
            result["layers"][layer][role] = {
                "mean_sample_mse": float(np.mean([v["mse"] for v in values])),
                "pooled_relative_mse": sum(v["squared_error"] for v in values)
                                       / max(sum(v["target_energy"] for v in values), 1e-20),
            }
    return result


def paired_difference(reference, candidate, seed):
    if [r["sample_id"] for r in reference] != [r["sample_id"] for r in candidate]:
        raise ValueError("Paired diagnostics require identical ordered samples.")
    indices = np.random.default_rng(seed).integers(len(reference), size=(2000, len(reference)))
    result = {}
    for slot in SID_SLOT_NAMES:
        result[slot] = {}
        for key in ("kl", "topk_retention"):
            delta = np.array([c["slots"][slot][key] - r["slots"][slot][key]
                              for r, c in zip(reference, candidate)])
            result[slot][key] = {"mean": float(delta.mean()),
                                 "ci95": np.quantile(delta[indices].mean(1), [0.025, 0.975]).tolist()}
    return result


def learned_state(train_block):
    return {
        "transform_state_dict": _tree_cpu(train_block.transforms.state_dict()),
        "activation_quantizer_state_dict": _tree_cpu(train_block.activation_quantizers.state_dict()),
        "lwc_parameters": {name: wrapper.lwc_state()
                           for name, wrapper in _flat_wrappers(train_block.block).items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--task", choices=("ad", "video", "product"), required=True)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--heldout_samples", type=int, default=128)
    parser.add_argument("--heldout_offset", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--diagnose_train", action="store_true",
                        help="Read completed arm checkpoints and diagnose training samples; no optimizer.")
    args = parser.parse_args()
    arms = args.arms.split(",")
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        raise ValueError("Unknown or duplicate arm.")
    if min(args.samples, args.heldout_samples, args.epochs) <= 0:
        raise ValueError("Sample and epoch counts must be positive.")
    if args.heldout_offset < max(512, args.samples):
        raise ValueError("Held-out must not overlap baseline final512 or pilot training.")
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    if args.diagnose_train:
        if not (args.output_dir / "complete.json").is_file():
            raise ValueError("Training diagnostics require a completed pilot directory.")
        if (args.output_dir / "train_summary.json").exists():
            raise ValueError("Training diagnostics already exist; refusing to overwrite.")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    split = default_calib_split(args.data_dir, "test", task_name=args.task)

    def data(count, offset):
        return load_task_data(task_name=args.task, tokenizer=tokenizer, data_dir=args.data_dir,
                              split=split, sample_size=count, sample_offset=offset,
                              require_answer=False, drop_final_assistant=True)

    train_data = data(args.samples, 0)
    heldout_data = data(args.heldout_samples, args.heldout_offset)
    if len(train_data) != args.samples or len(heldout_data) != args.heldout_samples:
        raise ValueError("Loader returned fewer samples than requested.")
    if set(train_data) & set(heldout_data):
        raise ValueError("Train/heldout sample IDs overlap.")

    def batches(samples):
        return build_lfq_sid_slot_batches(tokenizer=tokenizer, samples=list(samples.values()),
                                         prompt_token="<|sid_begin|>", device=torch.device("cpu"))

    complete = batches(train_data)
    heldout = batches(heldout_data)
    config = replace(_checkpoint_config(args.checkpoint_dir, args.layer),
                     epochs=args.epochs, train_sample_size=args.samples,
                     validation_sample_size=0, epoch_eval_interval=0, final_objective="mse",
                     learn_transform=True, use_lwc=True, use_lac=True, learn_lac=True)
    config.validate()
    if config.weight_group_size != 0:
        raise ValueError("This pilot expects per-output-channel baseline checkpoints.")
    protocol = {**vars(args), "checkpoint_dir": str(args.checkpoint_dir),
                "output_dir": str(args.output_dir.resolve()), "config": asdict(config),
                "train_ids": list(train_data), "heldout_ids": list(heldout_data),
                "token_sha256": hashlib.sha256(b"".join(
                    batch["input_ids"].numpy().tobytes() for batch in complete + heldout)).hexdigest(),
                "balanced_weights": {role: 0.25 for role in ROLES},
                "energy_normalization": False, "selection": "last_epoch",
                "note": "GT prefix, calibration heldout previously observed; not blind test or full recall."}
    if args.diagnose_train:
        saved = json.loads((args.output_dir / "protocol.json").read_text())
        for key in ("task", "model_path", "data_dir", "checkpoint_dir", "layer", "samples",
                    "heldout_samples", "heldout_offset", "epochs", "seed", "token_sha256"):
            if saved[key] != protocol[key]:
                raise ValueError(f"Training diagnostic protocol mismatch: {key}")
    else:
        write_json(args.output_dir / "protocol.json", protocol)
    print("[protocol]", json.dumps(protocol, ensure_ascii=False), flush=True)

    def model():
        return AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation="sdpa").to("cuda:0").eval().requires_grad_(False)

    teacher, student = model(), model()
    n_layers = len(student.model.layers)
    if not 0 <= args.layer < n_layers:
        raise ValueError("Selected layer is out of range.")
    restore_flatquant_core_layers_from_checkpoints(
        model=student, layer_indices=list(range(n_layers)), config=config,
        checkpoint_dir=args.checkpoint_dir, act_quant_mode="shared_input")
    baseline_block = student.model.layers[args.layer]
    checkpoint_path = args.checkpoint_dir / f"layer_{args.layer:02d}.pt"
    source_state = _load_flatquant_checkpoint(checkpoint_path, layer_idx=args.layer)
    slot_ids = {slot: torch.tensor(sid_slot_token_ids(tokenizer, slot), device="cuda:0")
                for slot in SID_SLOT_NAMES}
    measured_layers = sorted({max(0, args.layer - 1), args.layer, n_layers - 1})
    records, summaries = {}, {}
    eval_batches = complete if args.diagnose_train else heldout
    eval_ids = list(train_data if args.diagnose_train else heldout_data)
    output_prefix = "train_" if args.diagnose_train else ""

    def evaluate_arm(arm):
        rows = evaluate(teacher, student, eval_batches, eval_ids, slot_ids,
                        measured_layers, f"{'train' if args.diagnose_train else 'heldout'} {arm}")
        records[arm], summaries[arm] = rows, summarize(rows)
        write_json(args.output_dir / f"{output_prefix}{arm}_samples.json", rows)
        comparisons = {f"{arm}_minus_baseline": paired_difference(records["baseline"], rows, args.seed)
                       for arm, rows in records.items() if arm != "baseline"}
        for ref, candidate in (("prefill_mse", "complete_mse"), ("complete_mse", "balanced_mse")):
            if ref in records and candidate in records:
                comparisons[f"{candidate}_minus_{ref}"] = paired_difference(records[ref], records[candidate], args.seed)
        write_json(args.output_dir / f"{output_prefix}summary.json", {"arms": summaries, "paired_differences": comparisons})
        print(f"[{'train_diag' if args.diagnose_train else 'heldout'}] {arm}: " + " ".join(
            f"{slot}:KL={summaries[arm]['slots'][slot]['kl']:.6f},"
            f"top32={summaries[arm]['slots'][slot]['topk_retention']:.4%}"
            for slot in SID_SLOT_NAMES), flush=True)

    evaluate_arm("baseline")
    if args.diagnose_train:
        for arm in arms:
            path = args.output_dir / arm / f"layer_{args.layer:02d}.pt"
            state = _load_flatquant_checkpoint(path, layer_idx=args.layer)
            if state.get("stage_calibration") != arm or state["config"] != asdict(config):
                raise ValueError(f"Arm checkpoint protocol mismatch: {path}")
            restored = _TrainableFlatQuantBlock(copy.deepcopy(teacher.model.layers[args.layer]),
                                               config=config, act_quant_mode="shared_input")
            _restore_flatquant_train_state(restored, state, checkpoint_path=path)
            student.model.layers[args.layer] = _finalize_flat_block(restored)[0].eval()
            del restored, state
            evaluate_arm(arm)
            student.model.layers[args.layer] = baseline_block
        write_json(args.output_dir / "train_complete.json", {"arms": arms, "samples": len(complete)})
        return
    streams = {}
    for arm in arms:
        coverage = "prefill" if arm == "prefill_mse" else "complete"
        if coverage not in streams:
            selected_batches = [prefill_batch(batch) for batch in complete] if coverage == "prefill" else complete
            streams[coverage] = (
                capture_inputs(teacher, args.layer, selected_batches, f"capture FP {coverage}"),
                capture_inputs(student, args.layer, selected_batches, f"capture quant {coverage}"),
            )
        # Every arm starts from EXACTLY the same learned block, with a new AdamW state.
        set_seed(args.seed)
        train_block = _TrainableFlatQuantBlock(copy.deepcopy(teacher.model.layers[args.layer]),
                                              config=config, act_quant_mode="shared_input")
        _restore_flatquant_train_state(train_block, source_state, checkpoint_path=checkpoint_path)
        print(f"[train] arm={arm} layer={args.layer} objective={'stage_mean' if arm == 'balanced_mse' else 'token_mean'}", flush=True)
        initial, final, _, _, best_epoch, metrics = _train_flat_block(
            teacher_block=teacher.model.layers[args.layer], train_block=train_block,
            fp_inputs=streams[coverage][0], quant_inputs=streams[coverage][1],
            config=config, layer_idx=args.layer,
            reconstruction_loss=balanced_mse if arm == "balanced_mse" else F.mse_loss)
        training = {"arm": arm, "initial_objective": initial, "final_objective": final,
                    "best_epoch": best_epoch, "epochs": [asdict(metric) for metric in metrics],
                    "note": "MSE log fields denote this arm's reconstruction reduction; compare within arm."}
        write_json(args.output_dir / f"{arm}_training.json", training)
        arm_dir = args.output_dir / arm
        arm_dir.mkdir()
        # Store only the changed block; other blocks remain at checkpoint_dir.
        state = {**dict(source_state), **learned_state(train_block), "config": asdict(config),
                 "initial_loss": initial, "final_loss": final, "initial_mse_loss": initial,
                 "final_mse_loss": final, "initial_validation_mse_loss": None,
                 "final_validation_mse_loss": None, "best_epoch": None,
                 "epoch_metrics": training["epochs"], "stage_calibration": arm,
                 "initialization_checkpoint": str(checkpoint_path)}
        torch.save(state, arm_dir / f"layer_{args.layer:02d}.pt")
        student.model.layers[args.layer] = _finalize_flat_block(train_block)[0].eval()
        del train_block, state
        evaluate_arm(arm)
        student.model.layers[args.layer] = baseline_block
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output_dir / "complete.json", {"arms": arms, "samples": len(heldout), "layer": args.layer})
    print(f"[complete] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
