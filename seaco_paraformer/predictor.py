"""
CIF Predictor V3：从 encoder hidden 预测 token 数量并生成 acoustic embeddings。

抽取自 FunASR 训练版本，包含原始 cif 和向量化 cif_v1（用于 ONNX 导出）。
默认 cif 使用 for 循环（PT 推理用），向量化版本用于 ONNX 导出。
"""

import torch
import torch.nn as nn

from .layers import LayerNorm
from .utils import make_pad_mask


def cif(hidden, alphas, threshold):
    """原始 CIF 实现（含 for 循环，PT 推理用）。

    输出 acoustic_embeds 按 batch 内最大 token 数 pad。
    """
    batch_size, len_time, hidden_size = hidden.size()
    integrate = torch.zeros([batch_size], device=hidden.device)
    frame = torch.zeros([batch_size, hidden_size], device=hidden.device)
    list_fires = []
    list_frames = []

    for t in range(len_time):
        alpha = alphas[:, t]
        distribution_completion = torch.ones([batch_size], device=hidden.device) - integrate
        integrate += alpha
        list_fires.append(integrate)
        fire_place = integrate >= threshold
        integrate = torch.where(
            fire_place, integrate - torch.ones([batch_size], device=hidden.device), integrate
        )
        cur = torch.where(fire_place, distribution_completion, alpha)
        remainds = alpha - cur
        frame += cur[:, None] * hidden[:, t, :]
        list_frames.append(frame)
        frame = torch.where(
            fire_place[:, None].repeat(1, hidden_size),
            remainds[:, None] * hidden[:, t, :],
            frame,
        )

    fires = torch.stack(list_fires, 1)
    frames = torch.stack(list_frames, 1)
    list_ls = []
    # 用 int64（long）避免与 tensor.size()/索引（int64）混用，
    # 否则整体导出 ONNX 时下游 Concat 会绑定 int32+int64 报类型错。
    len_labels = torch.round(alphas.sum(-1)).long()
    max_label_len = len_labels.max()
    for b in range(batch_size):
        fire = fires[b, :]
        l = torch.index_select(
            frames[b, :, :], 0, torch.nonzero(fire >= threshold).squeeze(-1)
        )
        pad_l = torch.zeros(
            [max_label_len - l.size(0), hidden_size], device=hidden.device
        )
        list_ls.append(torch.cat([l, pad_l], 0))
    return torch.stack(list_ls, 0), fires


def cif_v1_export(hidden: torch.Tensor, alphas: torch.Tensor, threshold: float) -> tuple:
    """向量化 CIF（无 for 循环，用于 ONNX 导出，TRT 兼容）。

    软边界分配版（与 PT cif() for-loop 数学等价）：
        每帧 alpha 沿时间累积，跨越 threshold 整数倍边界即 fire 一个 token。
        跨边界的帧，其 alpha 按比例拆分到边界前后两个相邻 token（cur/remainds），
        与 PT cif() 完全一致。

    背景：旧实现用 floor(cum/thr) one-hot 把整帧硬分配给单个 token，丢失了
        边界帧的拆分。长输入下影响被平均掉，但短/截断输入（边界帧占比大）
        会导致 acoustic_embeds 偏差累积 → 后段 token 声学表示错乱 → 解码乱码。
        软分配版消除该偏差，所有长度与 PT 一致。

    向量化实现（区间重叠分配 + bmm，无 Loop/scatter，支持任意 alpha，TRT 最稳）：
        cum = cumsum(alphas)；帧 t 占累积区间 [cum[t]-alpha[t], cum[t])
        token j 占累积区间 [j*thr, (j+1)*thr)
        帧 t 对 token j 权重 = 两区间重叠 = clamp(min(cur,(j+1)thr)-max(prev,j*thr), >=0)
        与 PT cif() 的 cur/remainds 拆分数学一致；alpha>thr 跨多 token 时也自动正确
    """
    b, t, d = hidden.size()
    cum_alphas = torch.cumsum(alphas, dim=1)  # (b, t)
    floor_cum = torch.floor(cum_alphas / threshold)

    # token 数量（不变）
    token_num = floor_cum[:, -1].long()
    max_token_num = t

    # peak 标记（不变）
    floor_diff = torch.zeros_like(floor_cum)
    floor_diff[:, 1:] = floor_cum[:, 1:] - floor_cum[:, :-1]
    floor_diff[:, 0] = floor_cum[:, 0]
    cif_peak = (floor_diff > 0).float()

    # ---- 软边界分配（通用版：帧累积区间与 token 累积区间的重叠长度，支持任意 alpha） ----
    # token j 占累积区间 [j*thr, (j+1)*thr)；帧 t 占 [cum[t]-alpha[t], cum[t]) = [prev, cur)。
    # 帧 t 对 token j 的权重 = 两区间重叠长度 = clamp(min(cur,(j+1)*thr) - max(prev, j*thr), >=0)。
    # 该定义与 PT cif() 的 cur/remainds 拆分数学一致，且 alpha>thr 跨多 token 时自动正确分配
    # （中间被完整跨越的 token 拿到整段 thr），不依赖"单帧至多跨一界"假设。
    # 全程逐元素 min/max/sub/clamp + bmm，无 Loop/scatter，ONNX/TRT 最稳。
    cur = cum_alphas                                  # (b, t) 帧终点累积值
    prev = cum_alphas - alphas                        # (b, t) 帧起点累积值
    slot = torch.arange(max_token_num, device=hidden.device, dtype=hidden.dtype)  # (K,)
    lo = slot * threshold                             # (K,) 各 token 区间下界 j*thr
    hi = (slot + 1.0) * threshold                     # (K,) 各 token 区间上界 (j+1)*thr
    # 广播到 (b, t, K)
    cur_e = cur.unsqueeze(-1)
    prev_e = prev.unsqueeze(-1)
    overlap = torch.minimum(cur_e, hi) - torch.maximum(prev_e, lo)  # (b, t, K)
    assign = torch.clamp(overlap, min=0.0)            # 负重叠（无交集）置 0
    acoustic_embeds = torch.bmm(assign.transpose(1, 2), hidden)  # (b, K, d)

    return acoustic_embeds, cif_peak


def _cif_v1_export_legacy(hidden: torch.Tensor, alphas: torch.Tensor, threshold: float) -> tuple:
    """旧版硬分配 CIF（floor one-hot），保留作对照。

    ⚠ 已知缺陷：边界帧整帧硬分配给单个 token，短/截断输入下 acoustic 偏差累积致解码乱码。
    已被软分配版 cif_v1_export 替代，仅供回归对照，勿用于生产导出。
    """
    b, t, d = hidden.size()
    cum_alphas = torch.cumsum(alphas, dim=1)
    floor_cum = torch.floor(cum_alphas / threshold)
    token_num = floor_cum[:, -1].long()
    max_token_num = t
    floor_diff = torch.zeros_like(floor_cum)
    floor_diff[:, 1:] = floor_cum[:, 1:] - floor_cum[:, :-1]
    floor_diff[:, 0] = floor_cum[:, 0]
    cif_peak = (floor_diff > 0).float()
    token_idx = (floor_cum - 1).long().clamp(min=0, max=max_token_num - 1)
    assign = torch.zeros(b, t, max_token_num, device=hidden.device, dtype=hidden.dtype)
    assign.scatter_(2, token_idx.unsqueeze(-1), 1.0)
    weighted_assign = assign * alphas.unsqueeze(-1)
    acoustic_embeds = torch.bmm(weighted_assign.transpose(1, 2), hidden)
    return acoustic_embeds, cif_peak





class CifPredictorV3(nn.Module):
    """CIF Predictor V3（含上采样 cnn_blstm 时间戳预测）。

    配置固定：
        upsample_type = cnn_blstm
        use_cif1_cnn = False
        tail_threshold = 0.45
    """

    def __init__(
        self,
        idim,
        l_order,
        r_order,
        threshold=1.0,
        dropout=0.1,
        smooth_factor=1.0,
        noise_threshold=0,
        tail_threshold=0.0,
        smooth_factor2=1.0,
        noise_threshold2=0,
        upsample_times=5,
        **kwargs,
    ):
        super().__init__()
        self.pad = nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = nn.Conv1d(idim, idim, l_order + r_order + 1)
        self.cif_output = nn.Linear(idim, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.threshold = threshold
        self.smooth_factor = smooth_factor
        self.noise_threshold = noise_threshold
        self.tail_threshold = tail_threshold
        self.upsample_times = upsample_times

        # cnn_blstm 上采样
        self.upsample_cnn = nn.ConvTranspose1d(idim, idim, upsample_times, upsample_times)
        self.blstm = nn.LSTM(
            idim, idim, 1, bias=True, batch_first=True, dropout=0.0, bidirectional=True
        )
        self.cif_output2 = nn.Linear(idim * 2, 1)

        self.smooth_factor2 = smooth_factor2
        self.noise_threshold2 = noise_threshold2

    def forward(self, hidden, target_label=None, mask=None, ignore_id=-1,
                mask_chunk_predictor=None, target_label_length=None):
        h = hidden
        context = h.transpose(1, 2)
        queries = self.pad(context)
        output = torch.relu(self.cif_conv1d(queries))

        # alphas2 (timestamp head)：use_cif1_cnn=False → 用 context
        _output = context
        output2 = self.upsample_cnn(_output)
        output2 = output2.transpose(1, 2)
        output2, (_, _) = self.blstm(output2)
        alphas2 = torch.sigmoid(self.cif_output2(output2))
        alphas2 = torch.nn.functional.relu(alphas2 * self.smooth_factor2 - self.noise_threshold2)

        if mask is not None:
            mask2 = (
                mask.repeat(1, self.upsample_times, 1)
                .transpose(-1, -2)
                .reshape(alphas2.shape[0], -1)
            )
            mask2 = mask2.unsqueeze(-1)
            alphas2 = alphas2 * mask2
        alphas2 = alphas2.squeeze(-1)
        token_num2 = alphas2.sum(-1)

        # alphas (main head)
        output = output.transpose(1, 2)
        output = self.cif_output(output)
        alphas = torch.sigmoid(output)
        alphas = torch.nn.functional.relu(alphas * self.smooth_factor - self.noise_threshold)

        if mask is not None:
            mask = mask.transpose(-1, -2).float()
            alphas = alphas * mask
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor
        alphas = alphas.squeeze(-1)
        if mask is not None:
            mask = mask.squeeze(-1)

        if target_label_length is not None:
            target_length = target_label_length
        elif target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None

        token_num = alphas.sum(-1)

        if target_length is not None:
            alphas *= (target_length / token_num)[:, None].repeat(1, alphas.size(1))
        elif self.tail_threshold > 0.0:
            hidden, alphas, token_num = self.tail_process_fn(hidden, alphas, token_num, mask=mask)

        acoustic_embeds, cif_peak = cif(hidden, alphas, self.threshold)
        if target_length is None and self.tail_threshold > 0.0:
            token_num_int = torch.max(token_num).type(torch.int32).item()
            acoustic_embeds = acoustic_embeds[:, :token_num_int, :]
        return acoustic_embeds, token_num, alphas, cif_peak, token_num2

    def tail_process_fn(self, hidden, alphas, token_num=None, mask=None):
        b, t, d = hidden.size()
        tail_threshold = self.tail_threshold
        if mask is not None:
            zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
            ones_t = torch.ones_like(zeros_t)
            mask_1 = torch.cat([mask, zeros_t], dim=1)
            mask_2 = torch.cat([ones_t, mask], dim=1)
            mask = mask_2 - mask_1
            tail_threshold = mask * tail_threshold
            alphas = torch.cat([alphas, zeros_t], dim=1)
            alphas = torch.add(alphas, tail_threshold)
        else:
            tail_threshold = torch.tensor([tail_threshold], dtype=alphas.dtype).to(alphas.device)
            tail_threshold = torch.reshape(tail_threshold, (1, 1))
            alphas = torch.cat([alphas, tail_threshold], dim=1)

        zeros = torch.zeros((b, 1, d), dtype=hidden.dtype).to(hidden.device)
        hidden = torch.cat([hidden, zeros], dim=1)
        token_num = alphas.sum(dim=-1)
        token_num_floor = torch.floor(token_num)
        return hidden, alphas, token_num_floor


# ============================================================
# 模块自检：对比 PT cif（for循环软分配）vs cif_v1_export（向量化软分配）
#   vs _cif_v1_export_legacy（旧硬分配），验证算法等价性与代码是否生效。
# 运行：python -m seaco_paraformer.predictor
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    b, t, d = 1, 125, 8
    thr = 1.0
    hidden = torch.randn(b, t, d)
    # 真实 CIF：alpha = relu(sigmoid(x)*smooth_factor - noise_threshold)，smooth=1,noise=0
    # → alpha = sigmoid(x) ∈ (0,1)，恒 < threshold=1.0（单帧绝不跨界）
    alphas = torch.sigmoid(torch.randn(b, t) * 2.0)
    print(f"alphas 范围: min={alphas.min():.4f}, max={alphas.max():.4f} (真实模型恒<1.0)")

    # PT 版（for 循环软分配）
    ac_pt, _ = cif(hidden, alphas, thr)
    n_pt = int(torch.round(alphas.sum(-1)).item())

    # 新版（向量化软分配）
    ac_new, _ = cif_v1_export(hidden, alphas, thr)

    # 旧版（硬分配）
    ac_old, _ = _cif_v1_export_legacy(hidden, alphas, thr)

    print(f"token_num(PT round sum) = {n_pt}")
    print(f"ac_pt  shape={tuple(ac_pt.shape)}")
    print(f"ac_new shape={tuple(ac_new.shape)}")
    print(f"ac_old shape={tuple(ac_old.shape)}")

    n = min(n_pt, ac_pt.shape[1], ac_new.shape[1], ac_old.shape[1])
    # 对齐前 n 个 token 比较（PT 与新版应接近，旧版应有明显偏差）
    diff_new = (ac_pt[:, :n] - ac_new[:, :n]).abs().max().item()
    diff_old = (ac_pt[:, :n] - ac_old[:, :n]).abs().max().item()
    print(f"\n前 {n} 个 token acoustic_embeds 与 PT 的最大绝对误差：")
    print(f"  新版(软分配) vs PT : {diff_new:.6f}  {'✓ 接近' if diff_new < 1e-3 else '✗ 偏差大'}")
    print(f"  旧版(硬分配) vs PT : {diff_old:.6f}  {'✓ 接近' if diff_old < 1e-3 else '✗ 偏差大'}")
    print(f"\n新旧两版是否相同: {'是（代码可能未生效！）' if (ac_new[:, :n]-ac_old[:, :n]).abs().max().item() < 1e-9 else '否（代码已生效）'}")

    # ---- 关键：导出 cif_v1_export 为临时 ONNX，用 ORT 跑，验证导出链路是否保真 ----
    try:
        import onnxruntime as ort
        import tempfile, os as _os

        class _CifCore(torch.nn.Module):
            def forward(self, hidden, alphas):
                ac, _ = cif_v1_export(hidden, alphas, 1.0)
                return ac

        tmp = _os.path.join(tempfile.gettempdir(), "_cif_core_check.onnx")
        torch.onnx.export(
            _CifCore(), (hidden, alphas), tmp, opset_version=17,
            input_names=["hidden", "alphas"], output_names=["ac"],
            dynamic_axes={"hidden": {0: "b", 1: "t"}, "alphas": {0: "b", 1: "t"}, "ac": {0: "b", 1: "k"}},
        )
        sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
        ac_ort = sess.run(["ac"], {"hidden": hidden.numpy(), "alphas": alphas.numpy()})[0]
        ac_ort_t = torch.from_numpy(ac_ort)
        n2 = min(n, ac_ort_t.shape[1])
        diff_ort = (ac_pt[:, :n2] - ac_ort_t[:, :n2]).abs().max().item()
        print(f"\n[ONNX导出链路验证] ORT(cif_v1_export导出) vs PT 前{n2}token 最大误差: {diff_ort:.6f}"
              f"  {'✓ 导出保真' if diff_ort < 1e-3 else '✗ 导出后偏差大（ONNX算子语义问题）'}")
    except Exception as e:
        print(f"\n[ONNX导出链路验证] 跳过（{e}）")

