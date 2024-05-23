# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

# FAdam (Fisher Adam): an implentation in PyTorch of the paper:
#"FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher information"
# https://www.arxiv.org/abs/2405.12807


import torch
from torch.optim.optimizer import Optimizer


class FAdam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        c = 1,
        p = 0.5,
        eps=1e-15,
        momentum_dtype=torch.float32,
        fim_dtype=torch.float32,

    ):
        """
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): learning rate (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-15)
            TODO - explain c and p

            # Usage
            TODO
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            momentum_dtype=momentum_dtype,
            fim_dtype=variance_dtype,
            c = c,
            p=p,

        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        if closure is not None:
            with torch.enable_grad():
                # to fix linter, we do not keep the returned loss for use atm.
                closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            c = group["c"],
            p = group["p"],
            momentum_dtype = group["momentum_dtype"]
            fim_dtype = group["variance_dtype"]



            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError(
                        "FAdam does not support sparse gradients"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)

                    # momentum - EMA of gradient values
                    state["momentum"] = torch.zeros_like(
                        p,
                        dtype=momentum_dtype,
                    )

                    # variance uncentered - EMA of squared gradient values
                    state["fim"] = torch.ones_like(
                        p,
                        dtype=fim_dtype,
                    )


                # main processing -------------------------

                # update the steps for each param group update
                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                grad = p.grad

                # weight decay, AdamW style
                if weight_decay:
                    p.data.mul_(1 - lr * weight_decay)

                # update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # update uncentered variance
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # adjust using bias1
                bias_correction1 = 1 - beta1**step

                step_size = lr / bias_correction1

                # adjust using bias2
                denom_correction = (1 - beta2**step) ** 0.5  # avoids math import

                #if not use_numerical_guarantee:
                centered_variance = exp_avg_sq.sqrt() / denom_correction.add_(
                    1e-12, alpha=1
                )

                if use_numerical_guarantee:
                #    denom_max = max(denom_correction, eps)
                #    centered_variance = exp_avg_sq.sqrt() / denom_max

                    safe_variance = torch.clamp(centered_variance, min=1e-7)
                    safe_variance = safe_variance.sqrt()

                # lr update to compensation
                if use_kahan_summation:
                    compensation = state["compensation"]
                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)
                    # update weights with compensation (Kahan summation)
                    # save error back to compensation for next iteration
                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    compensation.add_(temp_buffer.sub_(p.data))

                elif use_numerical_guarantee:
                    p.data.addcdiv_(exp_avg, safe_variance, value=-step_size)
                else:
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)
