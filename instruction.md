# 出台阶响应辨识与分段预测

燃气轮机旁路温控器要换标定系数。试车台已经留下了一组阶跃响应采样和后续要出的预测时刻，值夜班的控制工程师需要把 `/app/stiff_ode_balance.py` 修到能直接交给测试台。现有程序对这段双模态响应的计算还不稳定。

探针记录是一份 UTF-8 JSON，顶层正好六个键：`asset_id`、`probe_step_s`、`probe_values`、`prediction_times_s`、`checkpoint_chunk`、`fit_tolerance`。

`asset_id` 是长度至多 64 的标签；空白不能出现在两端，控制字符一律不接受。`probe_step_s` 是探针相邻两次采样的秒数，大于 0 且不超过 10。`probe_values` 给 8 到 32 个正数，等间隔采样，反映冷态突然加热后的无量纲温升；这组曲线由两个衰减时间尺度不同的正指数模态叠成，没有测量噪声。`prediction_times_s` 给 1 到 200 个不重复、严格递增的非负数，单位是秒，最多到 3600。一次调用最多补算的预测点由 `checkpoint_chunk` 指定；含 1 和 200 在内，超出范围即整单退回，布尔值同样不接受。`fit_tolerance` 是允许的最大相对拟合误差，取 1e-9 到 1e-2。

上述以数字接收的字段都把布尔值视作类型错误。

试车台只会给出三个位置参数，顺序如下：

```bash
python3 /app/stiff_ode_balance.py <探针记录> <预测结果> <检查点>
```

若位置参数数量不符，进程返回 2，标准输出为空，错误流仅为 `运行拒收: USAGE`。

预测结果不展开排版。顶层键按以下顺序写入：

```text
status
asset_id
rate_slow_s
rate_fast_s
amplitude_slow
amplitude_fast
probe_fit_max_rel_error
predictions
```

文件仅以 LF 结尾。`status` 在本次调用算完全部预测时写 `complete`，否则写 `paused`。两个速率都取正数，慢速率小于快速率；两个幅值也是正数。相对拟合误差用全部探针中的最大绝对残差除以 `max(1, 最大探针读数)`。`predictions` 按请求到的先后列出，每一项只有 `t_s` 与 `value` 两个键，值均为有限数字。程序成功运行时 stdout 和 stderr 都不写内容。

两个模态要还原的是采样里真实存在的模型，不能拿一个指数近似了事。若探针序列无法用正幅值、正速率的两模态模型在给定容差内解释，就按模型不成立拒收。

## 分段跑的时候

首跑建立检查点，之后从尚未补算的预测点继续。检查点只出现下列四个字段，不增加其它键：

```text
input_sha256
coefficients
next_index
predictions
```

`input_sha256` 标识的是模型而非运行批次：先构造一个去掉 `checkpoint_chunk` 的五项对象，按 `sort_keys=True`、`separators=(",", ":")`、`ensure_ascii=False` 编码，再以 UTF-8 字节计算 SHA-256 的小写十六进制。`coefficients` 正好四个键：`amplitude_slow`、`rate_slow_s`、`amplitude_fast`、`rate_fast_s`。`next_index` 记录尚待补算的请求位置；`predictions` 则保留此前已经产出的结果。

检查点的摘要、键集合、数值类型、系数或预测前缀与实际请求对不上，就按检查点损坏拒收。已经完成的请求再次调用时，结果和检查点都保持原字节。因此，一次算完与分成多次补算，最终两份文件必须逐字节一致；暂停时已经写出的前缀也必须与最终结果同开头。

## 拒收和文件保护

输入结构、类型、范围或时间顺序有问题时，标准错误恰好是 `运行拒收: BAD_INPUT`，返回码为 1。两个正指数模型无法在容差内拟合时，标准错误为 `运行拒收: BAD_MODEL`，返回码为 1。目标路径重复、父目录不存在、目标是目录或不可写时，标准错误为 `运行拒收: IO_FAILURE`，返回码为 2。检查点损坏时，标准错误为 `运行拒收: BAD_CHECKPOINT`，返回码为 1。

拒收不能带 Python traceback。任何拒收都要保住调用前的预测结果和检查点字节；成功发布也不能留下临时文件。
