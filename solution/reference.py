#!/usr/bin/env python3
"""Two-mode exponential identification with resumable deterministic output."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path


INPUT_KEYS = {
    "asset_id",
    "probe_step_s",
    "probe_values",
    "prediction_times_s",
    "checkpoint_chunk",
    "fit_tolerance",
}
CHECKPOINT_KEYS = {"input_sha256", "coefficients", "next_index", "predictions"}
COEFFICIENT_KEYS = {"amplitude_slow", "rate_slow_s", "amplitude_fast", "rate_fast_s"}


class Refusal(Exception):
    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


def refuse(kind: str) -> None:
    sys.stderr.write(f"运行拒收: {kind}\n")
    raise SystemExit(1 if kind in {"BAD_INPUT", "BAD_MODEL", "BAD_CHECKPOINT"} else 2)


def pairs_hook(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Refusal("BAD_INPUT")
        result[key] = value
    return result


def read_json(path: Path, *, checkpoint: bool = False):
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_hook)
    except (OSError, UnicodeError, ValueError, Refusal):
        raise Refusal("BAD_CHECKPOINT" if checkpoint else "BAD_INPUT")
    return raw, document


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_input(document):
    if not isinstance(document, dict) or set(document) != INPUT_KEYS:
        raise Refusal("BAD_INPUT")
    asset_id = document["asset_id"]
    if (
        not isinstance(asset_id, str)
        or not asset_id
        or len(asset_id) > 64
        or asset_id != asset_id.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in asset_id)
    ):
        raise Refusal("BAD_INPUT")
    step = document["probe_step_s"]
    if not finite_number(step) or not 0 < step <= 10:
        raise Refusal("BAD_INPUT")
    values = document["probe_values"]
    if not isinstance(values, list) or not 8 <= len(values) <= 32:
        raise Refusal("BAD_INPUT")
    if any(not finite_number(value) or value <= 0 for value in values):
        raise Refusal("BAD_INPUT")
    times = document["prediction_times_s"]
    if not isinstance(times, list) or not 1 <= len(times) <= 200:
        raise Refusal("BAD_INPUT")
    if any(not finite_number(value) or value < 0 or value > 3600 for value in times):
        raise Refusal("BAD_INPUT")
    if any(left >= right for left, right in zip(times, times[1:])):
        raise Refusal("BAD_INPUT")
    chunk = document["checkpoint_chunk"]
    if isinstance(chunk, bool) or not isinstance(chunk, int) or not 1 <= chunk <= 200:
        raise Refusal("BAD_INPUT")
    tolerance = document["fit_tolerance"]
    if not finite_number(tolerance) or not 1e-9 <= tolerance <= 1e-2:
        raise Refusal("BAD_INPUT")
    return {
        "asset_id": asset_id,
        "step": float(step),
        "values": [float(value) for value in values],
        "times": [float(value) for value in times],
        "chunk": chunk,
        "tolerance": float(tolerance),
    }


def identify_two_modes(record):
    values = record["values"]
    step = record["step"]
    s11 = s12 = s22 = b1 = b2 = 0.0
    for index in range(len(values) - 2):
        previous = values[index]
        current = values[index + 1]
        following = values[index + 2]
        s11 += current * current
        s12 += current * previous
        s22 += previous * previous
        b1 += following * current
        b2 += following * previous
    determinant = s11 * s22 - s12 * s12
    if determinant <= 1e-24:
        raise Refusal("BAD_MODEL")
    recurrence_next = (b1 * s22 - b2 * s12) / determinant
    recurrence_previous = (s11 * b2 - s12 * b1) / determinant
    discriminant = recurrence_next * recurrence_next + 4.0 * recurrence_previous
    if discriminant <= 0:
        raise Refusal("BAD_MODEL")
    root = math.sqrt(discriminant)
    roots = [(recurrence_next + root) / 2.0, (recurrence_next - root) / 2.0]
    if any(not 0 < value < 1 for value in roots):
        raise Refusal("BAD_MODEL")
    first, second = roots
    denominator = first - second
    if abs(denominator) < 1e-15:
        raise Refusal("BAD_MODEL")
    amplitude_first = (values[1] - values[0] * second) / denominator
    amplitude_second = values[0] - amplitude_first
    if amplitude_first <= 0 or amplitude_second <= 0:
        raise Refusal("BAD_MODEL")
    rate_first = -math.log(first) / step
    rate_second = -math.log(second) / step

    modes = [
        (amplitude_first, rate_first),
        (amplitude_second, rate_second),
    ]
    modes.sort(key=lambda item: item[1])
    if modes[0][1] == modes[1][1]:
        raise Refusal("BAD_MODEL")
    residual = max(
        abs(
            values[index]
            - (
                modes[0][0] * math.exp(-modes[0][1] * index * step)
                + modes[1][0] * math.exp(-modes[1][1] * index * step)
            )
        )
        for index in range(len(values))
    )
    relative = residual / max(1.0, max(values))
    if relative > record["tolerance"]:
        raise Refusal("BAD_MODEL")
    return {
        "amplitude_slow": modes[0][0],
        "rate_slow_s": modes[0][1],
        "amplitude_fast": modes[1][0],
        "rate_fast_s": modes[1][1],
        "probe_fit_max_rel_error": relative,
    }


def predict(coefficients, when):
    return coefficients["amplitude_slow"] * math.exp(
        -coefficients["rate_slow_s"] * when
    ) + coefficients["amplitude_fast"] * math.exp(-coefficients["rate_fast_s"] * when)


def prediction_entry(coefficients, when):
    return {"t_s": when, "value": predict(coefficients, when)}


def validate_checkpoint(document, digest, coefficients, times):
    if not isinstance(document, dict) or set(document) != CHECKPOINT_KEYS:
        raise Refusal("BAD_CHECKPOINT")
    if document["input_sha256"] != digest:
        raise Refusal("BAD_CHECKPOINT")
    saved = document["coefficients"]
    if not isinstance(saved, dict) or set(saved) != COEFFICIENT_KEYS:
        raise Refusal("BAD_CHECKPOINT")
    for key in COEFFICIENT_KEYS:
        if not finite_number(saved[key]) or saved[key] <= 0:
            raise Refusal("BAD_CHECKPOINT")
        expected = coefficients[key]
        if abs(saved[key] - expected) > 1e-12 * max(1.0, abs(expected)):
            raise Refusal("BAD_CHECKPOINT")
    next_index = document["next_index"]
    if isinstance(next_index, bool) or not isinstance(next_index, int) or not 0 <= next_index <= len(times):
        raise Refusal("BAD_CHECKPOINT")
    predictions = document["predictions"]
    if not isinstance(predictions, list) or len(predictions) != next_index:
        raise Refusal("BAD_CHECKPOINT")
    for index, item in enumerate(predictions):
        if not isinstance(item, dict) or set(item) != {"t_s", "value"}:
            raise Refusal("BAD_CHECKPOINT")
        expected = prediction_entry(coefficients, times[index])
        if not finite_number(item["t_s"]) or item["t_s"] != expected["t_s"]:
            raise Refusal("BAD_CHECKPOINT")
        if not finite_number(item["value"]) or item["value"] != expected["value"]:
            raise Refusal("BAD_CHECKPOINT")
    return next_index, predictions


def compact_bytes(document):
    return (json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def validate_targets(output: Path, checkpoint: Path):
    if output == checkpoint or output.is_dir() or checkpoint.is_dir():
        raise Refusal("IO_FAILURE")
    for path in (output, checkpoint):
        if not path.parent.is_dir() or not os.access(path.parent, os.W_OK):
            raise Refusal("IO_FAILURE")
        if path.exists() and not os.access(path, os.W_OK):
            raise Refusal("IO_FAILURE")


def publish(output: Path, checkpoint: Path, output_bytes: bytes, checkpoint_bytes: bytes):
    handles = []
    staged = []
    old = {}
    try:
        for target, payload in ((output, output_bytes), (checkpoint, checkpoint_bytes)):
            old[target] = target.read_bytes() if target.exists() else None
            descriptor, name = tempfile.mkstemp(prefix=".stiff-", dir=target.parent)
            handles.append(descriptor)
            stage = Path(name)
            staged.append(stage)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(staged[0], output)
        os.replace(staged[1], checkpoint)
        staged.clear()
    except OSError:
        for target, previous in old.items():
            try:
                if previous is None:
                    target.unlink(missing_ok=True)
                else:
                    target.write_bytes(previous)
            except OSError:
                pass
        raise Refusal("IO_FAILURE")
    finally:
        for stage in staged:
            stage.unlink(missing_ok=True)


def main(argv):
    if len(argv) != 4:
        sys.stderr.write("运行拒收: USAGE\n")
        return 2
    source = Path(argv[1])
    output = Path(argv[2])
    checkpoint = Path(argv[3])
    try:
        raw, input_document = read_json(source)
        record = validate_input(input_document)
        coefficients_full = identify_two_modes(record)
        coefficients = {
            "amplitude_slow": coefficients_full["amplitude_slow"],
            "rate_slow_s": coefficients_full["rate_slow_s"],
            "amplitude_fast": coefficients_full["amplitude_fast"],
            "rate_fast_s": coefficients_full["rate_fast_s"],
        }
        validate_targets(output, checkpoint)
        digest_payload = {
            key: input_document[key]
            for key in INPUT_KEYS
            if key != "checkpoint_chunk"
        }
        digest = hashlib.sha256(
            json.dumps(
                digest_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        if checkpoint.exists() and checkpoint.stat().st_size:
            _, checkpoint_document = read_json(checkpoint, checkpoint=True)
            next_index, predictions = validate_checkpoint(
                checkpoint_document, digest, coefficients, record["times"]
            )
        else:
            next_index, predictions = 0, []
        stop = min(len(record["times"]), next_index + record["chunk"])
        predictions = list(predictions)
        for index in range(next_index, stop):
            predictions.append(prediction_entry(coefficients, record["times"][index]))
        complete = stop == len(record["times"])
        result = {
            "status": "complete" if complete else "paused",
            "asset_id": record["asset_id"],
            "rate_slow_s": coefficients["rate_slow_s"],
            "rate_fast_s": coefficients["rate_fast_s"],
            "amplitude_slow": coefficients["amplitude_slow"],
            "amplitude_fast": coefficients["amplitude_fast"],
            "probe_fit_max_rel_error": coefficients_full["probe_fit_max_rel_error"],
            "predictions": predictions,
        }
        new_checkpoint = {
            "input_sha256": digest,
            "coefficients": coefficients,
            "next_index": stop,
            "predictions": predictions,
        }
        publish(output, checkpoint, compact_bytes(result), compact_bytes(new_checkpoint))
        return 0
    except Refusal as exc:
        refuse(exc.kind)
    except (OSError, UnicodeError, ValueError):
        refuse("BAD_INPUT")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
