from asteroid.engine.schedulers import *
import numpy as np


class ExponentialWarmup(BaseScheduler):
    """ Scheduler to apply ramp-up during training to the learning rate.
    Args:
        optimizer: torch.optimizer.Optimizer, the optimizer from which to rampup the value from
        max_lr: float, the maximum learning to use at the end of ramp-up.
        rampup_length: int, the length of the rampup (number of steps).
        exponent: float, the exponent to be used.
    """

    def __init__(self, optimizer, max_lr, rampup_length, exponent=-5.0):
        super().__init__(optimizer)
        self.rampup_len = rampup_length
        self.max_lr = max_lr
        self.step_num = 1
        self.exponent = exponent

    def _get_scaling_factor(self):

        if self.rampup_len == 0:
            return 1.0
        else:

            current = np.clip(self.step_num, 0.0, self.rampup_len)
            phase = 1.0 - current / self.rampup_len
            return float(np.exp(self.exponent * phase * phase))

    def _get_lr(self):
        return self.max_lr * self._get_scaling_factor()


class WarmupCosineScheduler(BaseScheduler):
    """Warm up to ``max_lr`` and then decay smoothly with a cosine schedule.

    The public ``_get_scaling_factor`` intentionally remains the warmup factor
    only, so existing consistency-weight logic keeps the previous behavior:
    ramp up during warmup and stay saturated afterwards.
    """

    def __init__(
        self,
        optimizer,
        max_lr,
        rampup_length,
        total_steps,
        min_lr=1e-6,
        exponent=-5.0,
    ):
        super().__init__(optimizer)
        self.rampup_len = rampup_length
        self.max_lr = max_lr
        self.total_steps = max(total_steps, 1)
        self.min_lr = min_lr
        self.step_num = 1
        self.exponent = exponent

    def _get_scaling_factor(self):
        if self.rampup_len == 0:
            return 1.0

        current = np.clip(self.step_num, 0.0, self.rampup_len)
        phase = 1.0 - current / self.rampup_len
        return float(np.exp(self.exponent * phase * phase))

    def _get_lr(self):
        if self.rampup_len == 0 or self.step_num <= self.rampup_len:
            return self.max_lr * self._get_scaling_factor()

        if self.total_steps <= self.rampup_len:
            return self.max_lr

        decay_step = min(self.step_num, self.total_steps)
        progress = (decay_step - self.rampup_len) / (self.total_steps - self.rampup_len)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(self.min_lr + (self.max_lr - self.min_lr) * cosine)
