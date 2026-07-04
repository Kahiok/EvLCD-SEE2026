from absl.logging import info
from torch import nn
from torch.nn.modules.loss import _Loss

from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELB
from see.losses.image_loss import (
    ColorConstancyRegularization,
    DistillationSupervision,
    EdgeAuxLoss,
    ExposureControlRegularization,
    GradientLoss,
    HVILoss,
    IlluminationSmoothnessRegularization,
    L1CharbonnierLoss,
    LogBrightnessLoss,
    SEEMoreSampleConstraint,
    SpatialConsistencyLoss,
    ToneCurveAdjustmentLoss,
)


def get_single_loss(config):
    if config.NAME == "l1_charbonnier_loss":
        return L1CharbonnierLoss()
    elif config.NAME == "gradient_loss":
        return GradientLoss()
    elif config.NAME == "ssr_reconstruction_ssrloss":
        return L1CharbonnierLoss()
    elif config.NAME == "spatial_consistency_selfconstraints":
        return SpatialConsistencyLoss()
    elif config.NAME == "color_constancy_regularization":
        return ColorConstancyRegularization()
    elif config.NAME == "exposure_control_regularization":
        return ExposureControlRegularization(config.smoothing_kernal_size, config.expected_exposure_mean)
    elif config.NAME == "illumination_smoothness_regularization":
        return IlluminationSmoothnessRegularization()
    elif config.NAME == "see_net_more-sample-constraints":
        return SEEMoreSampleConstraint(config)
    elif config.NAME == "edge_aux_supervision":
        return EdgeAuxLoss()
    elif config.NAME == "distillation_supervision":
        return DistillationSupervision()
    elif config.NAME == "log_brightness_loss":
        return LogBrightnessLoss()
    elif config.NAME == "tone_curve_adjustment_supervision":
        return ToneCurveAdjustmentLoss(config)
    elif config.NAME == "hvi_loss":
        return HVILoss(k=getattr(config, 'k', 2))
    else:
        raise ValueError(f"Unknown loss: {config.NAME}")


class EventLowLightBatchLoss(_Loss):
    def __init__(self, configs):
        super(EventLowLightBatchLoss, self).__init__()
        self.loss_or_regularization = configs.NAME.lower().split("_")[-1]
        self.loss = get_single_loss(configs)

    def forward(self, batch):
        if self.loss_or_regularization == "loss":
            return self.loss(batch[ELB.NL], batch[ELB.PRD])
        elif self.loss_or_regularization == "selfconstraints":
            return self.loss(batch[ELB.LL], batch[ELB.PRD])
        elif self.loss_or_regularization == "regularization":
            return self.loss(batch[ELB.PRD])
        elif self.loss_or_regularization == "supervision":
            return self.loss(batch)
        elif self.loss_or_regularization == "more-sample-constraints":
            return self.loss(batch[ELB.NL], batch[ELB.SSR], batch[ELB.LL], batch[ELB.NLR], batch[ELB.NLR_EP])
        elif self.loss_or_regularization == "ssrloss":
            # SSR: model should reconstruct input when given input mean as prompt
            if ELB.SSR in batch and isinstance(batch[ELB.SSR], __import__('torch').Tensor):
                return self.loss(batch[ELB.LL], batch[ELB.SSR])
            return batch[ELB.PRD].new_zeros(1).squeeze()


class MixedLoss(_Loss):
    def __init__(self, configs):
        super(MixedLoss, self).__init__()
        self.loss = []
        self.weight = []
        self.criterion = nn.ModuleList()
        for item in configs:
            self.loss.append(item.NAME)
            self.weight.append(item.WEIGHT)
            self.criterion.append(EventLowLightBatchLoss(item))
        info(f"Init Mixed Loss: {configs}")

    def forward(self, batch):
        name_to_loss = []
        total = 0
        for n, w, fun in zip(self.loss, self.weight, self.criterion):
            tmp = fun(batch)
            name_to_loss.append((n, tmp))
            total = total + tmp * w
        return total, name_to_loss


class TaskWeightedMixedLoss(MixedLoss):
    """MixedLoss with per-task loss weighting.

    task_weights: e.g. {"low-normal": 2.0, "high-normal": 2.0, "normal-normal": 0.5}
    Loss for each mapping type is scaled by its weight, then normalized by the
    sum of weights so the total loss magnitude stays comparable to unweighted.
    """

    def __init__(self, configs, task_weights):
        super().__init__(configs)
        self.task_weights = task_weights
        info(f"TaskWeightedMixedLoss task_weights: {task_weights}")

    def forward(self, batch):
        import torch
        lqet = batch[ELB.LQET]   # list of B strings
        B = len(lqet)

        # Group batch indices by mapping type
        task_indices = {}
        for i, t in enumerate(lqet):
            task_indices.setdefault(t, []).append(i)

        total = 0.0
        name_acc = {}
        weight_sum = 0.0

        for task, indices in task_indices.items():
            w = self.task_weights.get(task, 1.0)

            # Build sub-batch for this mapping type
            sub = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == B:
                    sub[k] = v[indices]
                elif isinstance(v, (list, tuple)) and len(v) == B:
                    sub[k] = [v[i] for i in indices]
                else:
                    sub[k] = v

            loss_val, ntl = super().forward(sub)
            total = total + w * loss_val
            weight_sum += w
            for n, l in ntl:
                name_acc[n] = name_acc.get(n, 0.0) + w * l

        # Normalize so effective loss scale is unchanged
        total = total / weight_sum
        name_to_loss = [(n, v / weight_sum) for n, v in name_acc.items()]
        return total, name_to_loss
