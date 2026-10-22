#!/usr/bin/env python3
"""Legacy single-exponential response predictor."""

import json
import math
import sys
from pathlib import Path


def main(argv):
    if len(argv) != 4:
        sys.stderr.write("运行拒收: USAGE\n")
        return 2
    source, output, checkpoint = map(Path, argv[1:])
    record = json.loads(source.read_text(encoding="utf-8"))
    step = record["probe_step_s"]
    values = record["probe_values"]
    rate = -math.log(values[1] / values[0]) / step
    predictions = []
    for when in record["prediction_times_s"][: record["checkpoint_chunk"]]:
        predictions.append({"t_s": when, "value": values[0] * math.exp(-rate * when)})
    document = {
        "status": "paused",
        "asset_id": record["asset_id"],
        "rate_slow_s": rate,
        "rate_fast_s": rate,
        "amplitude_slow": values[0],
        "amplitude_fast": 0.0,
        "probe_fit_max_rel_error": 0.0,
        "predictions": predictions,
    }
    output.write_text(json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8")
    checkpoint.write_text(
        json.dumps({"next_index": len(predictions), "predictions": predictions}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
