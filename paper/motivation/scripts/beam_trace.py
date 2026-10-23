"""Observe the installed HF beam search without reimplementing its selection.

Validated on transformers 4.57.6. A/B are live beams; C is the actual returned
list, because max_new_tokens finishes every candidate at the last step.
"""
import inspect
from contextlib import contextmanager


@contextmanager
def trace_beams(model, prompt_length):
    names = ("_get_top_k_continuations", "_get_running_beams_for_next_iteration")
    if not all(hasattr(model, name) for name in names):
        raise RuntimeError("Beam tracing requires HF's new beam API (tested: transformers 4.57.6)")
    originals = {name: getattr(model, name) for name in names}
    local = {name: model.__dict__.get(name) for name in names}
    had_local = {name: name in model.__dict__ for name in names}
    signature = inspect.signature(originals[names[0]])
    if not {"cur_len", "decoder_prompt_len"}.issubset(signature.parameters):
        raise RuntimeError("Unsupported beam API signature; refusing an approximate trace")
    trace = {"live": {}, "expanded": {}, "early_stopped_candidates": 0}
    state = {}

    def top(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        if bound["decoder_prompt_len"] != prompt_length or bound["batch_size"] != 1:
            raise ValueError("Beam trace expects one unpadded prompt")
        depth = int(bound["cur_len"]) - prompt_length + 1
        state["depth"] = depth
        result = originals[names[0]](*args, **kwargs)
        scores, paths, _ = result
        trace["expanded"][str(depth)] = {
            "paths": paths[0, :, prompt_length:prompt_length+depth].detach().cpu().tolist(),
            "scores": scores[0].detach().float().cpu().tolist()}
        return result

    running_sig = inspect.signature(originals[names[1]])

    def running(*args, **kwargs):
        bound = running_sig.bind(*args, **kwargs).arguments
        depth = state["depth"]
        if depth < 3:
            trace["early_stopped_candidates"] += int(bound["next_token_hits_stopping_criteria"].sum())
        result = originals[names[1]](*args, **kwargs)
        paths, scores, _ = result
        if depth < 3:
            valid = scores[0] > -1e8
            trace["live"][str(depth)] = {
                "paths": paths[0, valid, prompt_length:prompt_length+depth].detach().cpu().tolist(),
                "scores": scores[0, valid].detach().float().cpu().tolist()}
        return result

    setattr(model, names[0], top)
    setattr(model, names[1], running)
    try:
        yield trace
    finally:
        for name in names:
            if had_local[name]:
                setattr(model, name, local[name])
            else:
                delattr(model, name)
