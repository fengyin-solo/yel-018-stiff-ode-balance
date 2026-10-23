"""Acceptance checks for the two-mode exponential response predictor."""

import hashlib
import json
import math
import os
import subprocess
import sys

import pytest


ENTRY = os.environ.get("STIFF_ODE_ENTRY", "/app/stiff_ode_balance.py")


def response_record():
    slow_amplitude = 6.25
    fast_amplitude = 2.75
    slow_rate = 0.19
    fast_rate = 2.35
    step = 0.25
    values = [
        slow_amplitude * math.exp(-slow_rate * index * step)
        + fast_amplitude * math.exp(-fast_rate * index * step)
        for index in range(12)
    ]
    return {
        "asset_id": "BYPASS-TC-17",
        "probe_step_s": step,
        "probe_values": values,
        "prediction_times_s": [0.0, 0.5, 1.25, 3.0, 7.5, 16.0],
        "checkpoint_chunk": 20,
        "fit_tolerance": 1e-9,
    }


def coefficients():
    return {
        "amplitude_slow": 6.25,
        "rate_slow_s": 0.19,
        "amplitude_fast": 2.75,
        "rate_fast_s": 2.35,
    }


def exact_value(when):
    model = coefficients()
    return model["amplitude_slow"] * math.exp(
        -model["rate_slow_s"] * when
    ) + model["amplitude_fast"] * math.exp(-model["rate_fast_s"] * when)


def write_json(path, document, raw=None):
    if raw is None:
        raw = json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n"
    path.write_bytes(raw.encode("utf-8"))
    return path


def invoke(tmp_path, document, *, output_name="result.json", checkpoint_name="checkpoint.json", raw=None):
    source = write_json(tmp_path / "record.json", document, raw=raw)
    output = tmp_path / output_name
    checkpoint = tmp_path / checkpoint_name
    result = subprocess.run(
        [sys.executable, ENTRY, str(source), str(output), str(checkpoint)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, source, output, checkpoint


def read_result(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_to_completion(tmp_path, document, stem="complete"):
    source = write_json(tmp_path / "record.json", document)
    output = tmp_path / f"{stem}.json"
    checkpoint = tmp_path / f"{stem}-checkpoint.json"
    collected = []
    for _ in range(20):
        result = subprocess.run(
            [sys.executable, ENTRY, str(source), str(output), str(checkpoint)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        collected.append(result)
        if result.returncode != 0 or read_result(output)["status"] == "complete":
            break
    return collected, output, checkpoint


def test_usage_has_exact_prefix_and_status():
    """题面：「若位置参数数量不符，进程返回 2，标准输出为空，错误流仅为 `运行拒收: USAGE`。」"""
    result = subprocess.run([sys.executable, ENTRY, "only-one"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert result.stderr.startswith("运行拒收: USAGE")
    assert result.stdout == ""


def test_too_many_position_arguments_are_refused(tmp_path):
    """题面：「试车台只会给出三个位置参数，顺序如下：」。"""
    output = tmp_path / "unused-output.json"
    checkpoint = tmp_path / "unused-checkpoint.json"
    result = subprocess.run(
        [sys.executable, ENTRY, "one", "two", str(output), str(checkpoint), "four"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2 and result.stderr == "运行拒收: USAGE\n"
    assert not output.exists() and not checkpoint.exists()


def test_success_is_quiet(tmp_path):
    """题面：「程序成功运行时 stdout 和 stderr 都不写内容。」"""
    result, _, output, _ = invoke(tmp_path, response_record())
    assert result.returncode == 0 and result.stdout == "" and result.stderr == "" and output.exists()


def test_absent_top_level_key_is_refused(tmp_path):
    """题面：「输入结构、类型、范围或时间顺序有问题时，标准错误恰好是 `运行拒收: BAD_INPUT`，返回码为 1。」"""
    document = response_record()
    document.pop("fit_tolerance")
    result, _, output, checkpoint = invoke(tmp_path, document)
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n"
    assert not output.exists() and not checkpoint.exists()


def test_extra_top_level_key_is_refused(tmp_path):
    """题面：「探针记录是一份 UTF-8 JSON，顶层正好六个键」。"""
    document = response_record()
    document["operator"] = "night"
    result, _, output, checkpoint = invoke(tmp_path, document)
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n"
    assert not output.exists() and not checkpoint.exists()


def test_duplicate_json_key_is_refused(tmp_path):
    """题面：「输入结构、类型、范围或时间顺序有问题时，标准错误恰好是 `运行拒收: BAD_INPUT`，返回码为 1。」"""
    raw = (
        '{"asset_id":"A","probe_step_s":0.25,"probe_values":[1,2,3,4,5,6,7,8],'
        '"prediction_times_s":[0],"checkpoint_chunk":1,"fit_tolerance":0.001,'
        '"fit_tolerance":0.002}'
    )
    result, _, output, checkpoint = invoke(tmp_path, None, raw=raw)
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n"
    assert not output.exists() and not checkpoint.exists()


@pytest.mark.parametrize(
    "asset_id",
    ["", " leading", "trailing ", "line\nbreak", "X" * 65],
)
def test_asset_id_shape_is_strict(tmp_path, asset_id):
    """题面：「`asset_id` 是长度至多 64 的标签；空白不能出现在两端，控制字符一律不接受。」"""
    document = response_record()
    document["asset_id"] = asset_id
    result, _, output, _ = invoke(tmp_path, document)
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n" and not output.exists()


def test_eight_probe_values_are_accepted(tmp_path):
    """题面：「`probe_values` 给 8 到 24 个正数」。"""
    document = response_record()
    document["probe_values"] = document["probe_values"][:8]
    result, _, output, _ = invoke(tmp_path, document)
    longest = response_record()
    longest["probe_values"] = [
        exact_value(index * longest["probe_step_s"])
        for index in range(24)
    ]
    longest_result, _, longest_output, _ = invoke(
        tmp_path,
        longest,
        output_name="longest-result.json",
        checkpoint_name="longest-checkpoint.json",
    )
    assert result.returncode == 0 and read_result(output)["status"] == "complete"
    assert longest_result.returncode == 0
    assert read_result(longest_output)["status"] == "complete"


def test_probe_step_upper_bound_is_accepted(tmp_path):
    """题面：「`probe_step_s` 是探针相邻两次采样的秒数，大于 0 且不超过 10。」"""
    document = response_record()
    document["probe_step_s"] = 10.0
    result, _, output, _ = invoke(tmp_path, document)
    assert result.returncode == 0 and output.is_file()


def test_seven_probe_values_are_refused(tmp_path):
    """题面：「`probe_values` 给 8 到 24 个正数」。"""
    document = response_record()
    document["probe_values"] = document["probe_values"][:7]
    result, _, output, _ = invoke(tmp_path, document)
    too_many = response_record()
    too_many["probe_values"] = [
        exact_value(index * too_many["probe_step_s"])
        for index in range(25)
    ]
    too_many_result, _, too_many_output, _ = invoke(
        tmp_path,
        too_many,
        output_name="too-many-result.json",
        checkpoint_name="too-many-checkpoint.json",
    )
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n"
    assert not output.exists()
    assert too_many_result.returncode == 1
    assert too_many_result.stderr == "运行拒收: BAD_INPUT\n"
    assert not too_many_output.exists()


def test_boolean_is_not_a_number(tmp_path):
    """题面：「上述以数字接收的字段都把布尔值视作类型错误。」"""
    document = response_record()
    document["probe_step_s"] = True
    result, _, output, _ = invoke(tmp_path, document)
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n" and not output.exists()


def test_prediction_times_must_be_strictly_increasing(tmp_path):
    """题面：「`prediction_times_s` 给 1 到 120 个不重复、严格递增的非负数」。"""
    document = response_record()
    document["prediction_times_s"] = [0.0, 1.0, 1.0]
    result, _, output, _ = invoke(tmp_path, document)
    longest = response_record()
    longest["prediction_times_s"] = [float(index) for index in range(120)]
    longest_result, _, longest_output, _ = invoke(
        tmp_path,
        longest,
        output_name="longest-predictions.json",
        checkpoint_name="longest-predictions-checkpoint.json",
    )
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n"
    assert not output.exists()
    assert longest_result.returncode == 0
    assert read_result(longest_output)["status"] == "paused"


def test_chunk_lower_bound_is_one(tmp_path):
    """题面：「一次调用最多补算的预测点由 `checkpoint_chunk` 指定；含 1 和 120 在内，超出范围即整单退回，布尔值同样不接受。」"""
    document = response_record()
    document["checkpoint_chunk"] = 1
    result, _, output, _ = invoke(tmp_path, document)
    assert result.returncode == 0 and read_result(output)["status"] == "paused"


def test_chunk_upper_bound_is_refused(tmp_path):
    """题面：「一次调用最多补算的预测点由 `checkpoint_chunk` 指定；含 1 和 120 在内，超出范围即整单退回，布尔值同样不接受。」"""
    document = response_record()
    document["checkpoint_chunk"] = 121
    result, _, output, _ = invoke(tmp_path, document)
    upper = response_record()
    upper["checkpoint_chunk"] = 120
    upper_result, _, upper_output, _ = invoke(
        tmp_path,
        upper,
        output_name="upper-result.json",
        checkpoint_name="upper-checkpoint.json",
    )
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_INPUT\n"
    assert not output.exists()
    assert upper_result.returncode == 0
    assert read_result(upper_output)["status"] == "complete"


def test_two_modes_are_recovered_to_tight_precision(tmp_path):
    """题面：「这组曲线由两个衰减时间尺度不同的正指数模态叠成，没有测量噪声。」"""
    result, _, output, _ = invoke(tmp_path, response_record())
    document = read_result(output)
    assert result.returncode == 0
    assert math.isclose(document["rate_slow_s"], 0.19, rel_tol=1e-9)
    assert math.isclose(document["rate_fast_s"], 2.35, rel_tol=1e-9)
    assert math.isclose(document["amplitude_slow"], 6.25, rel_tol=1e-9)
    assert math.isclose(document["amplitude_fast"], 2.75, rel_tol=1e-9)


def test_predictions_follow_the_recovered_model(tmp_path):
    """题面：「`predictions` 按请求到的先后列出，每一项只有 `t_s` 与 `value` 两个键，值均为有限数字。」"""
    result, _, output, _ = invoke(tmp_path, response_record())
    document = read_result(output)
    errors = [
        abs(item["value"] - exact_value(item["t_s"]))
        for item in document["predictions"]
    ]
    assert result.returncode == 0 and max(errors) <= 1e-11
    assert [item["t_s"] for item in document["predictions"]] == response_record()["prediction_times_s"]


def test_each_prediction_has_exact_keys(tmp_path):
    """题面：「文件仅以 LF 结尾。`status` 在本次调用算完全部预测时写 `complete`，否则写 `paused`。两个速率都取正数，慢速率小于快速率；两个幅值也是正数。相对拟合误差用全部探针中的最大绝对残差除以 `max(1, 最大探针读数)`。`predictions` 按请求到的先后列出，每一项只有 `t_s` 与 `value` 两个键，值均为有限数字。程序成功运行时 stdout 和 stderr 都不写内容。」"""
    result, _, output, _ = invoke(tmp_path, response_record())
    document = read_result(output)
    assert result.returncode == 0
    assert all(set(item) == {"t_s", "value"} for item in document["predictions"])


def test_each_prediction_is_finite(tmp_path):
    """题面：「文件仅以 LF 结尾。`status` 在本次调用算完全部预测时写 `complete`，否则写 `paused`。两个速率都取正数，慢速率小于快速率；两个幅值也是正数。相对拟合误差用全部探针中的最大绝对残差除以 `max(1, 最大探针读数)`。`predictions` 按请求到的先后列出，每一项只有 `t_s` 与 `value` 两个键，值均为有限数字。程序成功运行时 stdout 和 stderr 都不写内容。」"""
    result, _, output, _ = invoke(tmp_path, response_record())
    document = read_result(output)
    assert result.returncode == 0
    assert all(math.isfinite(item["t_s"]) and math.isfinite(item["value"]) for item in document["predictions"])


def test_two_modes_matter_for_late_prediction(tmp_path):
    """题面：「现有程序对这段双模态响应的计算还不稳定。」"""
    record = response_record()
    result, _, output, _ = invoke(tmp_path, record)
    document = read_result(output)
    values = record["probe_values"]
    naive_rate = -math.log(values[1] / values[0]) / record["probe_step_s"]
    naive_late = values[0] * math.exp(-naive_rate * 16.0)
    assert result.returncode == 0
    assert abs(document["predictions"][-1]["value"] - exact_value(16.0)) <= 1e-11
    assert abs(exact_value(16.0) - naive_late) > 0.2


def test_result_is_compact_ordered_and_newline_terminated(tmp_path):
    """题面：「预测结果不展开排版」"""
    result, _, output, _ = invoke(tmp_path, response_record())
    raw = output.read_bytes()
    document = json.loads(raw)
    assert result.returncode == 0
    assert list(document) == [
        "status",
        "asset_id",
        "rate_slow_s",
        "rate_fast_s",
        "amplitude_slow",
        "amplitude_fast",
        "probe_fit_max_rel_error",
        "predictions",
    ]
    assert b": " not in raw and raw.endswith(b"\n") and raw.count(b"\n") == 1


def test_single_exponential_measurements_fail_the_model_check(tmp_path):
    """题面：「若探针序列无法用正幅值、正速率的两模态模型在给定容差内解释，就按模型不成立拒收。」"""
    document = response_record()
    document["probe_values"] = [5.0 * math.exp(-0.4 * index * 0.25) for index in range(12)]
    document["fit_tolerance"] = 1e-9
    result, _, output, checkpoint = invoke(tmp_path, document)
    assert result.returncode == 1 and result.stderr == "运行拒收: BAD_MODEL\n"
    assert not output.exists() and not checkpoint.exists()


def test_paused_prefix_is_stable(tmp_path):
    """题面：「暂停时已经写出的前缀也必须与最终结果同开头。」"""
    one_shot = response_record()
    result, _, output, _ = invoke(tmp_path, one_shot, output_name="one.json", checkpoint_name="one.ckpt")
    one_document = read_result(output)
    document = response_record()
    document["checkpoint_chunk"] = 2
    paused_result, _, paused_output, _ = invoke(
        tmp_path, document, output_name="paused.json", checkpoint_name="paused.ckpt"
    )
    paused_document = read_result(paused_output)
    assert result.returncode == 0 and paused_result.returncode == 0
    assert paused_document["status"] == "paused"
    assert paused_document["predictions"] == one_document["predictions"][:2]


def test_chunked_resume_matches_one_shot_bytes(tmp_path):
    """题面：「一次算完与分成多次补算，最终两份文件必须逐字节一致」。"""
    one_shot_record = response_record()
    one_result, _, one_output, one_checkpoint = invoke(
        tmp_path, one_shot_record, output_name="one.json", checkpoint_name="one.ckpt"
    )
    chunked_record = response_record()
    chunked_record["checkpoint_chunk"] = 2
    runs, chunked_output, chunked_checkpoint = run_to_completion(tmp_path, chunked_record)
    assert one_result.returncode == 0 and all(item.returncode == 0 for item in runs) and len(runs) > 1
    assert chunked_output.read_bytes() == one_output.read_bytes()
    assert chunked_checkpoint.read_bytes() == one_checkpoint.read_bytes()


def test_completed_checkpoint_rerun_keeps_both_files(tmp_path):
    """题面：「已经完成的请求再次调用时，结果和检查点都保持原字节。」"""
    document = response_record()
    result, source, output, checkpoint = invoke(tmp_path, document)
    before_output = output.read_bytes()
    before_checkpoint = checkpoint.read_bytes()
    second = subprocess.run(
        [sys.executable, ENTRY, str(source), str(output), str(checkpoint)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0 and second.returncode == 0
    assert output.read_bytes() == before_output and checkpoint.read_bytes() == before_checkpoint


def test_checkpoint_digest_mismatch_is_refused(tmp_path):
    """题面：「检查点的摘要、键集合、数值类型、系数或预测前缀与实际请求对不上，就按检查点损坏拒收。」"""
    document = response_record()
    result, source, output, checkpoint = invoke(tmp_path, document)
    before_output = output.read_bytes()
    before_checkpoint = checkpoint.read_bytes()
    changed = response_record()
    changed["asset_id"] = "OTHER"
    write_json(source, changed)
    second = subprocess.run(
        [sys.executable, ENTRY, str(source), str(output), str(checkpoint)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0 and second.returncode == 1
    assert second.stderr == "运行拒收: BAD_CHECKPOINT\n"
    assert output.read_bytes() == before_output and checkpoint.read_bytes() == before_checkpoint


def test_checkpoint_is_lowercase_sha256(tmp_path):
    """题面：「检查点只出现下列四个字段，不增加其它键：」"""
    source = write_json(tmp_path / "record.json", response_record())
    output = tmp_path / "result.json"
    checkpoint = tmp_path / "checkpoint.json"
    result = subprocess.run(
        [sys.executable, ENTRY, str(source), str(output), str(checkpoint)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = response_record()
    payload.pop("checkpoint_chunk")
    expected = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert saved["input_sha256"] == expected


def test_duplicate_targets_are_refused(tmp_path):
    """题面：「目标路径重复、父目录不存在、目标是目录或不可写时」。"""
    source = write_json(tmp_path / "record.json", response_record())
    target = tmp_path / "same.json"
    result = subprocess.run(
        [sys.executable, ENTRY, str(source), str(target), str(target)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2 and result.stderr == "运行拒收: IO_FAILURE\n" and not target.exists()


def test_absent_parent_directory_is_refused(tmp_path):
    """题面：「目标路径重复、父目录不存在、目标是目录或不可写时」。"""
    source = write_json(tmp_path / "record.json", response_record())
    output = tmp_path / "unavailable-parent" / "result.json"
    checkpoint = tmp_path / "checkpoint.json"
    result = subprocess.run(
        [sys.executable, ENTRY, str(source), str(output), str(checkpoint)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2 and result.stderr == "运行拒收: IO_FAILURE\n" and not checkpoint.exists()


def test_refusal_preserves_existing_output_bytes(tmp_path):
    """题面：「任何拒收都要保住调用前的预测结果和检查点字节」。"""
    result, source, output, checkpoint = invoke(tmp_path, response_record())
    before_output = output.read_bytes()
    before_checkpoint = checkpoint.read_bytes()
    broken = response_record()
    broken["checkpoint_chunk"] = True
    write_json(source, broken)
    second = subprocess.run(
        [sys.executable, ENTRY, str(source), str(output), str(checkpoint)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0 and second.returncode == 1
    assert output.read_bytes() == before_output and checkpoint.read_bytes() == before_checkpoint
