"""Audit and evaluation utilities for the semantic record linkage tutorial."""

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import re

import litellm
import numpy as np
import pandas as pd
import lotus
from lotus.models import LM
from lotus.templates import task_instructions


PRIMARY_MODEL = "gpt-6-luna"
HISTORICAL_MODEL = "gpt-4.1-mini-2025-04-14"
PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"
PRICING_CHECKED_DATE = "2026-10-06"
PRIMARY_PRICES_PER_MILLION = {"input": 0.10, "output": 0.50}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def pair_grid(left, right):
    for side, frame in (("left", left), ("right", right)):
        if "unique_id" not in frame or frame.unique_id.isna().any() or not frame.unique_id.is_unique:
            raise ValueError(f"The {side} records require complete, unique record IDs.")
    return left[["unique_id"]].rename(columns={"unique_id": "left_id"}).merge(
        right[["unique_id"]].rename(columns={"unique_id": "right_id"}), how="cross"
    )


def label_pairs(pairs, truth):
    entity = truth.set_index("unique_id")["entity_id"]
    result = pairs.copy()
    a, b = result.left_id.map(entity), result.right_id.map(entity)
    if not a.notna().all() or not b.notna().all():
        raise ValueError("Every evaluated record requires a ground-truth entity ID.")
    result["is_match"] = a.eq(b)
    return result


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.dtype != bool or prediction.dtype != bool or truth.shape != prediction.shape:
        raise ValueError("Metrics require complete Boolean arrays of the same shape.")
    tp = int((truth & prediction).sum())
    fp = int((~truth & prediction).sum())
    fn = int((truth & ~prediction).sum())
    tn = int((~truth & ~prediction).sum())
    return dict(pairs=len(truth), TP=tp, FP=fp, FN=fn, TN=tn,
                precision=tp / (tp + fp) if tp + fp else np.nan,
                recall=tp / (tp + fn) if tp + fn else np.nan,
                F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan)


def select_threshold(validation):
    if not np.isfinite(validation.match_probability).all() or not validation.match_probability.between(0, 1).all():
        raise ValueError("Threshold selection requires finite probabilities between zero and one.")
    # Fixed grid; a tie chooses the higher threshold. Test labels are never used.
    thresholds = np.unique(np.r_[np.logspace(-6, -1, 26), np.linspace(.1, .99, 90), .999])
    rows = [dict(threshold=float(t), **metrics(validation.is_match,
            validation.match_probability >= t)) for t in thresholds]
    return pd.DataFrame(rows).sort_values(["F1", "threshold"], ascending=False).reset_index(drop=True)


def as_semantic_records(frame, fields):
    result = frame[["unique_id"]].copy()
    result["record"] = frame[fields].apply(
        lambda row: json.dumps({field: None if pd.isna(row[field]) else str(row[field])
                               for field in fields}, ensure_ascii=False), axis=1
    )
    return result.reset_index(drop=True)


def boolean_answer(output):
    if not isinstance(output, str):
        raise ValueError("LLM output must be text containing a strict Boolean.")
    match = re.fullmatch(r"\s*(?:Answer:\s*)?(True|False)\s*", output, re.IGNORECASE)
    if match is None:
        raise ValueError("LLM output was not a strict Boolean; the run is incomplete and must not be scored.")
    return match.group(1).lower() == "true"


class AuditedLM(LM):
    """Reject parser fallbacks and retain output/message hashes for each comparison."""

    def __init__(self, model=PRIMARY_MODEL, **kwargs):
        kwargs.setdefault("temperature", 0)
        kwargs.setdefault("max_tokens", 32)
        kwargs.setdefault("max_batch_size", 8)
        if model == PRIMARY_MODEL:
            kwargs.setdefault("reasoning_effort", "none")
            kwargs.setdefault("num_retries", 0)
        super().__init__(model=model, **kwargs)
        self.audit = []
        self.provider_usage = []
        self.audit_failed = False

    def _update_stats(self, response, is_cached=False):
        self.provider_usage.append(dict(
            model=response.model, usage=response.usage.model_dump(),
            finish_reason=response.choices[0].finish_reason, lotus_cached=is_cached,
        ))
        return super()._update_stats(response, is_cached)

    def _process_uncached_messages(self, uncached_data, all_kwargs, show_progress_bar, progress_bar_desc):
        # LOTUS 1.2.4 otherwise enables parameter dropping in its transport.
        # An unsupported setting must fail before it changes this experiment.
        if self.rate_limit is not None or self.tpm_limit is not None:
            raise ValueError("This bounded tutorial uses the fixed batch transport.")
        if litellm.drop_params:
            raise ValueError("Disable LiteLLM's global parameter dropping before this run.")
        return litellm.batch_completion(
            self.model, [message for message, _ in uncached_data],
            drop_params=False, max_workers=self.max_batch_size, **all_kwargs,
        )

    def __call__(self, messages, **kwargs):
        if self.audit_failed:
            raise ValueError("This run failed. Create a new AuditedLM before retrying.")
        try:
            fixed = generation_settings(self.model)
            if self.kwargs != fixed:
                raise ValueError("Actual generation settings differ from the fixed tutorial configuration.")
            for key, value in kwargs.items():
                if key in ("show_progress_bar", "progress_bar_desc"):
                    continue
                if key == "logprobs" and value is False:
                    continue
                if key in fixed and value == fixed[key]:
                    continue
                raise ValueError(f"Per-call setting {key!r} differs from the fixed tutorial configuration.")
            output = super().__call__(messages, **kwargs)
            if len(output.outputs) != len(messages):
                raise ValueError("Incomplete LLM response batch; no accuracy result can be computed.")
            for message, answer in zip(messages, output.outputs):
                item = dict(message_sha256=digest(message), raw_output=answer, decision=None)
                self.audit.append(item)
                try:
                    item["decision"] = boolean_answer(answer)
                except ValueError:
                    pass
            if any(x["decision"] is None for x in self.audit):
                raise ValueError("An invalid Boolean answer makes this run incomplete.")
            if any(x["finish_reason"] != "stop" for x in self.provider_usage):
                raise ValueError("A provider response did not finish normally; this run cannot be scored.")
        except Exception:
            self.audit_failed = True
            raise
        return output


def generation_settings(model):
    """Keep the historical replay unchanged; make current-model settings explicit."""
    if model not in {PRIMARY_MODEL, HISTORICAL_MODEL}:
        raise ValueError("Use the current primary model or the recorded historical baseline.")
    settings = {"temperature": 0, "max_completion_tokens": 32}
    if model == PRIMARY_MODEL:
        settings.update(reasoning_effort="none", num_retries=0)
    return settings


def run_spec(left, right, predicate, model=PRIMARY_MODEL):
    settings = generation_settings(model)
    spec = dict(model=model, temperature=0, max_tokens=32,
                lotus_version=version("lotus-ai"), litellm_version=version("litellm"),
                serialization_format=lotus.settings.serialization_format.value,
                predicate=predicate, left=left.to_dict("records"), right=right.to_dict("records"))
    if model == PRIMARY_MODEL:
        spec.update(reasoning_effort=settings["reasoning_effort"], num_retries=settings["num_retries"],
                    drop_params=False)
    return spec


def expected_messages(spec):
    """Rebuild the exact plain sem_join prompts without running a model."""
    if spec["lotus_version"] != "1.2.4" or version("lotus-ai") != "1.2.4":
        raise ValueError("Prompt regeneration is verified only for LOTUS 1.2.4.")
    if spec["litellm_version"] != version("litellm") or spec["temperature"] != 0 or spec["max_tokens"] != 32:
        raise ValueError("Package or generation settings differ from the fixed tutorial configuration.")
    fixed = generation_settings(spec["model"])
    if spec["model"] == PRIMARY_MODEL and any(spec.get(k) != fixed[k] for k in ("reasoning_effort", "num_retries")):
        raise ValueError("The primary run requires reasoning_effort='none' and num_retries=0.")
    if spec["model"] == PRIMARY_MODEL and spec.get("drop_params") is not False:
        raise ValueError("The primary run must reject unsupported API settings.")
    if spec["serialization_format"] != lotus.settings.serialization_format.value:
        raise ValueError("LOTUS serialization differs from the recorded run configuration.")
    if set(lotus.nl_expression.parse_cols(spec["predicate"])) != {"record:left", "record:right"}:
        raise ValueError("This tutorial expects a predicate over {record:left} and {record:right}.")
    left, right = pd.DataFrame(spec["left"]), pd.DataFrame(spec["right"])
    pair_grid(left, right)
    left_docs = task_instructions.df2multimodal_info(
        left[["record"]].rename(columns={"record": "record:left"}), ["record:left"]
    )
    right_docs = task_instructions.df2multimodal_info(
        right[["record"]].rename(columns={"record": "record:right"}), ["record:right"]
    )
    docs = task_instructions.merge_multimodal_info(left_docs, right_docs)
    # The model parameter is unused with strategy=None; this formats prompts only.
    return [task_instructions.filter_formatter(None, doc, spec["predicate"]) for doc in docs]


def expected_message_hashes(spec):
    return [digest(message) for message in expected_messages(spec)]


def preflight_cost(spec, budget_usd=0.10):
    """Allow cache-write input pricing and a margin; this is not an invoice cap."""
    if spec["model"] != PRIMARY_MODEL or not 0 < budget_usd <= 1:
        raise ValueError("The current-model demonstration requires a budget between $0 and $1.")
    messages = expected_messages(spec)
    if len(messages) > 225:
        raise ValueError("The structured demonstration allows at most 225 comparisons.")
    input_bound = sum(litellm.token_counter(model=spec["model"], messages=m) + 32 for m in messages)
    output_bound = len(messages) * spec["max_tokens"]
    prices = PRIMARY_PRICES_PER_MILLION
    upper = (input_bound * prices["input"] * 1.25 + output_bound * prices["output"]) / 1e6 * 1.2
    if upper > budget_usd:
        raise ValueError(f"Planned request estimate ${upper:.6f} exceeds budget ${budget_usd:.2f}.")
    return dict(budget_usd=budget_usd, planned_comparisons=len(messages),
                input_token_bound=input_bound, output_token_bound=output_bound,
                cache_write_input_multiplier=1.25, margin=1.2, estimated_upper_usd=upper,
                pricing_source=PRICING_SOURCE, pricing_checked_date=PRICING_CHECKED_DATE,
                prices_per_million=prices.copy())


def provider_summary(usage, spec, expected_count):
    if len(usage) != expected_count or any(
            x["finish_reason"] != "stop" or x.get("lotus_cached", False)
            or x["model"] != spec["model"]
            for x in usage):
        raise ValueError("Provider model, response status, or physical usage coverage is inconsistent.")
    input_tokens = sum(x["usage"]["prompt_tokens"] for x in usage)
    output_tokens = sum(x["usage"]["completion_tokens"] for x in usage)
    prices = PRIMARY_PRICES_PER_MILLION
    return dict(input_tokens=input_tokens, output_tokens=output_tokens,
                response_models=sorted({x["model"] for x in usage}),
                uncached_price_estimate_usd=(input_tokens * prices["input"] + output_tokens * prices["output"]) / 1e6,
                prices_per_million=prices.copy(), pricing_source=PRICING_SOURCE,
                pricing_checked_date=PRICING_CHECKED_DATE)


def validate_audit(audit, spec):
    hashes = expected_message_hashes(spec)
    if len(audit) != len(hashes):
        raise ValueError("Incomplete comparison audit; no accuracy result can be computed.")
    for item, expected_hash in zip(audit, hashes):
        if item["message_sha256"] != expected_hash:
            raise ValueError("Recorded messages differ from the expected input pairs or prompt formatting.")
        if type(item["decision"]) is not bool or boolean_answer(item["raw_output"]) != item["decision"]:
            raise ValueError("Recorded decisions disagree with the strict Boolean outputs.")


def save_snapshot(path, spec, lm, joined, seconds, preflight=None):
    if lm.audit_failed:
        raise ValueError("A failed run cannot be saved or scored.")
    if lm.model != spec["model"] or lm.kwargs != generation_settings(spec["model"]):
        raise ValueError("Actual model configuration differs from the declared run specification.")
    left, right = pd.DataFrame(spec["left"]), pd.DataFrame(spec["right"])
    pairs = pair_grid(left, right)
    validate_audit(lm.audit, spec)
    decisions = [item["decision"] for item in lm.audit]
    expected = set(map(tuple, pairs.loc[decisions, ["left_id", "right_id"]].to_numpy()))
    observed = set(map(tuple, joined[["unique_id:left", "unique_id:right"]].to_numpy()))
    if expected != observed or len(joined) != len(observed):
        raise ValueError("Comparison audit and LOTUS join results disagree or contain duplicate pairs.")
    snapshot = dict(schema=1, spec_sha256=digest(spec), spec=spec,
                    recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=seconds, stats=asdict(lm.stats),
                    scored_pairs=len(pairs), parse_failures=0,
                    pairs=pairs.to_dict("records"))
    # Build records explicitly so each prediction remains tied to its input pair.
    snapshot["pairs"] = [dict(**pair, **audit) for pair, audit in
                         zip(pairs.to_dict("records"), lm.audit)]
    snapshot["predictions_sha256"] = digest(snapshot["pairs"])
    if spec["model"] == PRIMARY_MODEL:
        snapshot.update(schema=2, provider_usage=lm.provider_usage,
                        provider_usage_sha256=digest(lm.provider_usage),
                        usage_summary=provider_summary(lm.provider_usage, spec, len(pairs)))
        if preflight is not None:
            snapshot["preflight"] = preflight
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A dated result is evidence: reruns must use a new path.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, indent=2) + "\n")
    return snapshot


def load_snapshot(path, spec):
    snapshot = json.loads(Path(path).read_text())
    if snapshot.get("schema") not in {1, 2} or snapshot.get("parse_failures") != 0:
        raise ValueError("Unsupported snapshot schema or an incomplete recorded run.")
    if snapshot["spec_sha256"] != digest(spec) or digest(snapshot["spec"]) != digest(spec):
        raise ValueError("Snapshot inputs, prompt, model or package version changed. Run live or restore the pinned environment.")
    if snapshot["predictions_sha256"] != digest(snapshot["pairs"]):
        raise ValueError("Snapshot predictions failed their integrity check.")
    validate_audit(snapshot["pairs"], spec)
    pairs = pd.DataFrame(snapshot["pairs"])
    expected = pair_grid(pd.DataFrame(spec["left"]), pd.DataFrame(spec["right"]))
    if not pairs[["left_id", "right_id"]].equals(expected):
        raise ValueError("Snapshot record pairs differ from the expected ordered Cartesian join.")
    if len(pairs) != snapshot["scored_pairs"]:
        raise ValueError("Snapshot comparison count is incomplete or inconsistent.")
    if spec["model"] == PRIMARY_MODEL:
        if snapshot["schema"] != 2 or digest(snapshot["provider_usage"]) != snapshot["provider_usage_sha256"]:
            raise ValueError("The current-model snapshot requires intact provider usage.")
        if provider_summary(snapshot["provider_usage"], spec, len(pairs)) != snapshot["usage_summary"]:
            raise ValueError("Recorded provider usage and the priced summary disagree.")
    return snapshot
