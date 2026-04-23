import torch

from desed_task.utils.schedulers import WarmupCosineScheduler


def _make_scheduler(decay_power):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    return WarmupCosineScheduler(
        optimizer,
        max_lr=1e-3,
        rampup_length=10,
        total_steps=110,
        min_lr=1e-5,
        decay_power=decay_power,
    )


def test_smaller_decay_power_drops_faster_after_peak():
    baseline = _make_scheduler(decay_power=1.0)
    faster = _make_scheduler(decay_power=0.6)

    baseline.step_num = 20
    faster.step_num = 20

    assert faster._get_lr() < baseline._get_lr()


def test_decay_power_one_preserves_standard_cosine_shape():
    scheduler = _make_scheduler(decay_power=1.0)
    scheduler.step_num = 60

    progress = (scheduler.step_num - scheduler.rampup_len) / (
        scheduler.total_steps - scheduler.rampup_len
    )
    expected = scheduler.min_lr + (scheduler.max_lr - scheduler.min_lr) * 0.5 * (
        1.0 + torch.cos(torch.tensor(torch.pi * progress))
    )

    assert torch.isclose(
        torch.tensor(scheduler._get_lr()),
        expected.float(),
        atol=1e-8,
    )
