# -*- coding: utf-8 -*-
"""
长尾检测损失统一模块（SCI 扩展版）

在 YOLO26 的 E2ELoss（one2many + one2one 两分支）上，把分类损失（BCE 项）
替换为不同的长尾损失，通过 `on_train_start` 回调注入（避免 YOLO.train() 重建
模型对象时丢弃瞬态 criterion 的问题）。

支持的 loss_type：
  bce      —— 原生 BCE（基线）
  focal    —— Focal Loss（单独，Lin et al. ICCV 2017）
  cb       —— Class-Balanced Loss（单独，Cui et al. CVPR 2019）
  cbfocal  —— CB-Focal（本文方法，CB 权重 + Focal 调制）
  seesaw   —— Seesaw Loss（Wang et al. CVPR 2021）

用法：
    from losses import apply_loss
    model = YOLO('weights/yolo26s.pt')
    apply_loss(model, loss_type='seesaw', samples_per_class=[69,86,802,3819])
    model.train(...)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import E2ELoss, v8DetectionLoss


# ---------------------------------------------------------------------------
# 1. Focal BCE
# ---------------------------------------------------------------------------
class FocalBCEWithLogits(nn.Module):
    """带 focal 调制的 BCEWithLogitsLoss（reduction=none），与 ultralytics 原生 FocalLoss 同公式。"""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        modulating = (1.0 - p_t) ** self.gamma
        return bce * modulating


# ---------------------------------------------------------------------------
# 2. Seesaw BCE（Wang et al. CVPR 2021 的 sigmoid 多标签版本）
# ---------------------------------------------------------------------------
class SeesawBCEWithLogits(nn.Module):
    """Seesaw Loss 的 sigmoid 版本。

    核心思想：当一个「头部类 j」作为负样本、而该 anchor 的正类是更稀有的类 i
    （N_j > N_i）时，用 (1-sigma_j)^p 下调 j 的负类梯度，从而缓解头类对稀有类
    的压制（mitigation）。p=2 为论文默认。

    参数 samples_per_class：各类别训练框数（按 names 顺序）。
    """

    def __init__(self, samples_per_class, p: float = 2.0, eps: float = 1e-7):
        super().__init__()
        self.register_buffer("freq", torch.tensor(samples_per_class, dtype=torch.float32))
        self.p = p
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sigma = torch.sigmoid(logits)
        pos = targets
        neg = 1.0 - targets

        # 每个 anchor 的正类（软标签取 argmax 作为唯一正类）
        pos_cls = targets.argmax(dim=-1)                 # (bs, A)
        N_pos = self.freq[pos_cls]                       # (bs, A)
        N_j = self.freq.view(1, 1, -1)                   # (1, 1, nc)
        has_pos = targets.sum(dim=-1, keepdim=True) > 0  # (bs, A, 1) 纯背景 anchor 不缓解

        # 负类 j 比正类更常见（N_j > N_pos）时，施加 mitigation
        mitigation = torch.where(
            (N_j > N_pos.unsqueeze(-1)) & has_pos,
            (1.0 - sigma) ** self.p,
            torch.ones_like(sigma),
        )

        pos_loss = -(pos * torch.log(sigma + self.eps))
        neg_loss = -(neg * torch.log(1.0 - sigma + self.eps)) * mitigation
        return pos_loss + neg_loss


# ---------------------------------------------------------------------------
# 3. Class-Balanced 权重（Cui et al. CVPR 2019）
# ---------------------------------------------------------------------------
def cb_weights(samples_per_class, beta: float = 0.9999, lam: float = 1.0):
    """有效样本数反比权重，归一化到均值为 1，返回形状 (1, 1, nc)。

    lam 为「平衡强度」指数（召回-精度权衡开关）：
      lam = 1   纯 CB
      lam > 1   更激进（稀有类进一步抬升，偏向高召回 / 筛查模式）
      lam < 1   更温和（偏向高精度 / 精确模式）
      lam = 0   退化为无加权（等价 BCE）
    """
    n = torch.tensor(samples_per_class, dtype=torch.float32)
    effective_num = (1.0 - beta ** n) / (1.0 - beta)
    w = (1.0 - beta) / effective_num
    w = w / w.mean()
    if lam != 1.0:
        w = w ** lam
        w = w / w.mean()
    return w.view(1, 1, -1)


# ---------------------------------------------------------------------------
# 4. 通用长尾检测损失（替换 v8DetectionLoss 的 bce / class_weights）
# ---------------------------------------------------------------------------
class LongTailDetectionLoss(v8DetectionLoss):
    """在 v8DetectionLoss 上按 loss_type 替换分类损失。"""

    def __init__(
        self,
        model,
        tal_topk: int = 10,
        tal_topk2=None,
        loss_type: str = "cbfocal",
        gamma: float = 2.0,
        beta: float = 0.9999,
        beta_start: float = None,
        beta_end: float = None,
        lam: float = 1.0,
        samples_per_class=None,
        seesaw_p: float = 2.0,
    ):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.loss_type = loss_type

        # 1) 替换 bce 项
        if loss_type in ("focal", "cbfocal"):
            self.bce = FocalBCEWithLogits(gamma=gamma)
        elif loss_type == "seesaw":
            assert samples_per_class is not None, "Seesaw 需要 samples_per_class"
            self.bce = SeesawBCEWithLogits(samples_per_class=samples_per_class, p=seesaw_p).to(self.device)
        else:  # bce / cb
            self.bce = nn.BCEWithLogitsLoss(reduction="none")

        # 2) class-balanced 权重（覆盖 super 读到的 model.class_weights）
        #    β 退火：训练进度 progress∈[0,1] 时 beta = beta_start + (beta_end-beta_start)*progress
        #    若不启用退火，beta_start=beta_end=beta，退化为静态权重。
        self.samples_per_class = samples_per_class
        self.beta_start = beta if beta_start is None else beta_start
        self.beta_end = beta if beta_end is None else beta_end
        self.lam = lam
        self.anneal = (self.beta_start != self.beta_end)

        if loss_type in ("cb", "cbfocal") and samples_per_class is not None:
            self.class_weights = cb_weights(samples_per_class, self.beta_start, lam).to(self.device)

    def set_progress(self, progress: float):
        """按训练进度 progress∈[0,1] 更新 class_weights（β 退火）。

        仅在启用退火且 loss_type 含 CB 时生效；progress 会被裁剪到 [0,1]。
        """
        if not self.anneal or self.samples_per_class is None:
            return
        progress = max(0.0, min(1.0, float(progress)))
        beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        self.class_weights = cb_weights(self.samples_per_class, beta, self.lam).to(self.device)


class LongTailE2ELoss(E2ELoss):
    """E2ELoss 的长尾版本：one2many / one2one 两分支均用 LongTailDetectionLoss。"""

    def __init__(
        self,
        model,
        loss_type: str = "cbfocal",
        gamma: float = 2.0,
        beta: float = 0.9999,
        beta_start: float = None,
        beta_end: float = None,
        lam: float = 1.0,
        samples_per_class=None,
        seesaw_p: float = 2.0,
    ):
        def _loss_fn(m, tal_topk=10, tal_topk2=None):
            return LongTailDetectionLoss(
                m,
                tal_topk=tal_topk,
                tal_topk2=tal_topk2,
                loss_type=loss_type,
                gamma=gamma,
                beta=beta,
                beta_start=beta_start,
                beta_end=beta_end,
                lam=lam,
                samples_per_class=samples_per_class,
                seesaw_p=seesaw_p,
            )

        super().__init__(model, loss_fn=_loss_fn)

    def set_progress(self, progress: float):
        """把训练进度转发给 one2many / one2one 两个分支（β 退火）。"""
        self.one2many.set_progress(progress)
        self.one2one.set_progress(progress)


def apply_loss(
    model,
    loss_type: str = "cbfocal",
    samples_per_class=None,
    gamma: float = 2.0,
    beta: float = 0.9999,
    beta_start: float = None,
    beta_end: float = None,
    lam: float = 1.0,
    seesaw_p: float = 2.0,
):
    """把指定长尾损失注入到 YOLO model 的 criterion。

    关键：`YOLO.train()` 会重建模型对象，criterion 是瞬态属性会被丢弃，
    必须用 `on_train_start` 回调在模型重建后、首个优化步前注入。

    可选优化：
      - β 退火：传 beta_start / beta_end（如 0.999 → 0.9999），训练过程按
        epoch 线性过渡，课程式加权；不传则退化为静态 beta。
      - λ 权衡开关：传 lam（>1 偏筛查高召回，<1 偏精确，=1 纯 CB）。
    """
    from ultralytics.utils.torch_utils import unwrap_model

    def _inject(trainer):
        m = unwrap_model(trainer.model)
        m.criterion = LongTailE2ELoss(
            m,
            loss_type=loss_type,
            gamma=gamma,
            beta=beta,
            beta_start=beta_start,
            beta_end=beta_end,
            lam=lam,
            samples_per_class=samples_per_class,
            seesaw_p=seesaw_p,
        )
        print(
            f"[loss] 已注入 criterion={type(m.criterion).__name__} "
            f"(loss_type={loss_type}, gamma={gamma}, beta={beta}, "
            f"beta_start={beta_start}, beta_end={beta_end}, lam={lam}, "
            f"samples={None if samples_per_class is None else list(samples_per_class)})"
        )

    model.add_callback("on_train_start", _inject)

    # β 退火：每个 epoch 开始前按进度更新 criterion 的 class_weights
    # （on_train_epoch_start 触发时 trainer.epoch 已赋值为当前 epoch，进度精确）
    if beta_start is not None and beta_end is not None and beta_start != beta_end:
        def _anneal(trainer):
            progress = trainer.epoch / max(trainer.epochs - 1, 1)
            m = unwrap_model(trainer.model)
            if hasattr(m, "criterion") and hasattr(m.criterion, "set_progress"):
                m.criterion.set_progress(progress)

        model.add_callback("on_train_epoch_start", _anneal)

    return model


# BRACOL REVIEWED 训练集各类别框数（按 names 顺序 Cercospora/Miner/Phoma/Rust）
BRACOL_SAMPLES_PER_CLASS = [133, 171, 1310, 4184]
