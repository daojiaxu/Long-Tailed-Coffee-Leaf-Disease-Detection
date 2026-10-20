# -*- coding: utf-8 -*-
"""
Unified long-tailed detection loss module

Replaces the classification (BCE) term of YOLO26's E2ELoss (one2many + one2one
branches) with different long-tailed losses, injected via an `on_train_start`
callback (avoids the discarded-transient-criterion problem when YOLO.train()
rebuilds the model object).

Supported loss_type values:
  bce      -- native BCE (baseline)
  focal    -- Focal Loss alone (Lin et al. ICCV 2017)
  cb       -- Class-Balanced Loss alone (Cui et al. CVPR 2019)
  cbfocal  -- CB-Focal (proposed in this work: CB weighting + Focal modulation)
  seesaw   -- Seesaw Loss (Wang et al. CVPR 2021)

Usage:
    from losses import apply_loss
    model = YOLO('weights/yolo26s.pt')
    apply_loss(model, loss_type='seesaw', samples_per_class=[133,171,1310,4184])
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
    """BCEWithLogitsLoss (reduction=none) with focal modulation; same formula as the native ultralytics FocalLoss."""

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
# 2. Seesaw BCE (sigmoid multi-label version of Wang et al. CVPR 2021)
# ---------------------------------------------------------------------------
class SeesawBCEWithLogits(nn.Module):
    """Sigmoid version of the Seesaw Loss.

    Core idea: when a head class j acts as a negative sample while the anchor's
    positive class is a rarer class i (N_j > N_i), the negative gradient of j is
    down-weighted by (1-sigma_j)^p, mitigating (mitigation) the suppression of
    rare classes by head classes. p=2 is the default of the paper.

    Parameter samples_per_class: per-class training box counts (in names order).
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

        # positive class of each anchor (argmax of the soft label as the single positive class)
        pos_cls = targets.argmax(dim=-1)                 # (bs, A)
        N_pos = self.freq[pos_cls]                       # (bs, A)
        N_j = self.freq.view(1, 1, -1)                   # (1, 1, nc)
        has_pos = targets.sum(dim=-1, keepdim=True) > 0  # (bs, A, 1) pure-background anchors are not mitigated

        # apply mitigation when negative class j is more common than the positive class (N_j > N_pos)
        mitigation = torch.where(
            (N_j > N_pos.unsqueeze(-1)) & has_pos,
            (1.0 - sigma) ** self.p,
            torch.ones_like(sigma),
        )

        pos_loss = -(pos * torch.log(sigma + self.eps))
        neg_loss = -(neg * torch.log(1.0 - sigma + self.eps)) * mitigation
        return pos_loss + neg_loss


# ---------------------------------------------------------------------------
# 3. Class-Balanced weighting (Cui et al. CVPR 2019)
# ---------------------------------------------------------------------------
def cb_weights(samples_per_class, beta: float = 0.9999, lam: float = 1.0):
    """Effective-number reciprocal weights, normalized to mean 1; returns shape (1, 1, nc).

    lam is the "balance strength" exponent (a recall-precision trade-off switch):
      lam = 1   pure CB
      lam > 1   more aggressive (rare classes lifted further; favors high recall / screening mode)
      lam < 1   milder (favors high precision / precision mode)
      lam = 0   degenerates to no weighting (equivalent to BCE)
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
# 4. General long-tailed detection loss (replaces the bce / class_weights of v8DetectionLoss)
# ---------------------------------------------------------------------------
class LongTailDetectionLoss(v8DetectionLoss):
    """Replaces the classification loss on v8DetectionLoss according to loss_type."""

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

        # 1) replace the bce term
        if loss_type in ("focal", "cbfocal"):
            self.bce = FocalBCEWithLogits(gamma=gamma)
        elif loss_type == "seesaw":
            assert samples_per_class is not None, "Seesaw requires samples_per_class"
            self.bce = SeesawBCEWithLogits(samples_per_class=samples_per_class, p=seesaw_p).to(self.device)
        else:  # bce / cb
            self.bce = nn.BCEWithLogitsLoss(reduction="none")

        # 2) class-balanced weights (override the model.class_weights read by super)
        #    beta annealing: at training progress progress in [0,1],
        #    beta = beta_start + (beta_end - beta_start) * progress
        #    if annealing is disabled, beta_start = beta_end = beta (static weights).
        self.samples_per_class = samples_per_class
        self.beta_start = beta if beta_start is None else beta_start
        self.beta_end = beta if beta_end is None else beta_end
        self.lam = lam
        self.anneal = (self.beta_start != self.beta_end)

        if loss_type in ("cb", "cbfocal") and samples_per_class is not None:
            self.class_weights = cb_weights(samples_per_class, self.beta_start, lam).to(self.device)

    def set_progress(self, progress: float):
        """Update class_weights according to the training progress (beta annealing).

        Only takes effect when annealing is enabled and loss_type contains CB;
        progress is clipped to [0,1].
        """
        if not self.anneal or self.samples_per_class is None:
            return
        progress = max(0.0, min(1.0, float(progress)))
        beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        self.class_weights = cb_weights(self.samples_per_class, beta, self.lam).to(self.device)


class LongTailE2ELoss(E2ELoss):
    """Long-tailed version of E2ELoss: both the one2many and one2one branches use LongTailDetectionLoss."""

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
        """Forwards the training progress to the one2many and one2one branches (beta annealing)."""
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
    """Inject the specified long-tailed loss into the criterion of a YOLO model.

    Key point: YOLO.train() rebuilds the model object, and criterion is a
    transient attribute that gets discarded; it must be injected via an
    `on_train_start` callback, after the model rebuild and before the first
    optimization step.

    Optional enhancements:
      - beta annealing: pass beta_start / beta_end (e.g. 0.999 -> 0.9999) for a
        curriculum weighting that transitions linearly over training; omit for a
        static beta.
      - lambda trade-off switch: pass lam (>1 favors high recall for screening,
        <1 favors high precision, =1 pure CB).
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
            f"[loss] injected criterion={type(m.criterion).__name__} "
            f"(loss_type={loss_type}, gamma={gamma}, beta={beta}, "
            f"beta_start={beta_start}, beta_end={beta_end}, lam={lam}, "
            f"samples={None if samples_per_class is None else list(samples_per_class)})"
        )

    model.add_callback("on_train_start", _inject)

    # beta annealing: update the criterion's class_weights by progress before
    # each epoch (when on_train_epoch_start fires, trainer.epoch already holds
    # the current epoch, so the progress is exact)
    if beta_start is not None and beta_end is not None and beta_start != beta_end:
        def _anneal(trainer):
            progress = trainer.epoch / max(trainer.epochs - 1, 1)
            m = unwrap_model(trainer.model)
            if hasattr(m, "criterion") and hasattr(m.criterion, "set_progress"):
                m.criterion.set_progress(progress)

        model.add_callback("on_train_epoch_start", _anneal)

    return model


# BRACOL REVIEWED per-class box counts (in names order: Cercospora/Miner/Phoma/Rust)
BRACOL_SAMPLES_PER_CLASS = [133, 171, 1310, 4184]
