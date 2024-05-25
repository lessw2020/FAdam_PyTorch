# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

# FAdam (Fisher Adam): an implentation in PyTorch of the paper:
# "FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher information"
# https://www.arxiv.org/abs/2405.12807


import torch
from torch.optim.optimizer import Optimizer
from typing import Tuple, Optional


class FAdam(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        clip: float = 1.0,
        p: float = 0.5,
        eps: float = 10e-8,
        eps2: float = 10e-4,
        momentum_dtype: torch.dtype = torch.float32,
        fim_dtype: torch.dtype = torch.float32,
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
            clip (float, optional): maximum norm of the gradient (default: 1.0)
            TODO - explain p

            # Usage
            TODO
        """
        
        # sanity checks
        assert len(betas) == 2, "betas must be a tuple of two floats"
        assert 0 <= betas[0] < 1 and 0 <= betas[1] < 1, "betas must be between 0 and 1"
        assert lr > 0, "lr must be non-negative"
        assert eps > 0, "eps must be non-negative"
        assert eps2 > 0, "eps2 must be non-negative"
        assert clip >= 0, "clip must be non-negative"
        
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            eps2=eps2,
            momentum_dtype=momentum_dtype,
            fim_dtype=fim_dtype,
            clip=clip,
            p=p,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                # to fix linter, we do not keep the returned loss for use atm.
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            eps2 = group["eps2"]
            clip = group["clip"]
            pval = group["p"]
            momentum_dtype = group["momentum_dtype"]
            fim_dtype = group["fim_dtype"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("FAdam does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state.setdefault("step", torch.tensor(0.0))
                    state.setdefault(
                        "momentum", torch.zeros_like(p, dtype=momentum_dtype)
                    )
                    state.setdefault("fim", torch.ones_like(p, dtype=fim_dtype))


                # main processing -------------------------

                # update the steps for each param group update
                state["step"] += 1
                step = state["step"]

                momentum = state["momentum"]
                fim = state["fim"]
                grad = p.grad

                # begin FAdam algo -------------------------
                # 6 - beta2 bias correction per Section 3.4.4
                curr_beta2 = beta2 * (1 - beta2 ** (step - 1)) / (1 - beta2**step)

                # 7 - update fim
                fim = (curr_beta2 * fim) + (1 - curr_beta2) * (grad * grad)

                # 8 - adaptive epsilon
                rms_grad = torch.sqrt(torch.mean((grad * grad)))
                curr_eps = min(eps, eps2 * rms_grad)

                # 8 - compute natural gradient
                fim_base = fim**pval + curr_eps  # **(2*pval)

                grad_nat = grad / fim_base

                # 9 - clip the natural gradient
                rms = torch.sqrt(torch.mean(grad_nat**2))
                divisor = max(1, rms)
                divisor = divisor / clip
                grad_nat = grad_nat / divisor

                # 10 - update momentum
                momentum.mul_(beta1).add_(grad_nat, alpha=1 - beta1)

                # 11 - weight decay
                grad_weights = p / fim_base

                # 12 - clip weight decay
                rms = torch.sqrt(torch.mean(grad_weights**2))
                divisor = max(1, rms)
                divisor /= clip
                grad_weights = grad_weights / divisor

                # 13 - compute update
                full_step = momentum + (weight_decay * grad_weights)
                lr_step = lr * full_step

                # 14 - update weights
                p.sub_(lr_step)

        return loss
