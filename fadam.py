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
try:
    from torchtitan.utils import logger
except:
    print("no logger")
    pass


class FAdam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay = 0.001,
        betas=(0.9, 0.999),
        clip = 1.0,
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
            clip (float, optional): maximum norm of the gradient (default: 1.0)
            TODO - explain p

            # Usage
            TODO
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            momentum_dtype=momentum_dtype,
            fim_dtype=fim_dtype,
            clip = clip,
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
            clip = group["clip"]
            pval = group["p"]
            momentum_dtype = group["momentum_dtype"]
            fim_dtype = group["fim_dtype"]
            weight_decay = group["weight_decay"]


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

                momentum = state["momentum"]
                fim = state["fim"]
                grad = p.grad

                # begin FAdam algo -------------------------
                #6 - beta2 bias correction per Section 3.4.4
                curr_beta2 = (beta2 *((1-beta2**step-1))/(1-beta2**step))

                #7 - update fim
                fim = (curr_beta2*fim) + (1-curr_beta2)*(grad*grad)

                #8 - compute natural gradient
                grad_nat = grad/(fim +eps)

                #9 - clip the natural gradient
                rms = torch.sqrt(torch.mean(grad_nat**2))
                divisor = max(1, rms)
                divisor = divisor/ clip
                grad_nat = grad_nat/divisor

                #10 - update momentum
                momentum.mul_(beta1).add_(grad, alpha=1-beta1)

                #11 - weight decay
                grad_weights = p/(fim+eps)

                #12 - clip weight decay
                rms = torch.sqrt(torch.mean(grad_weights**2))
                divisor = max(1,rms)
                divisor /= clip

                grad_weights = grad_weights/ divisor

                #13 - compute update
                full_step = momentum +(weight_decay*grad_weights)

                lr_step = lr * full_step
                lr_step *=10000
                #print(f"lr_step {lr_step}")

                #14 - update weights
                #print(f"pre-step {p.data}")
                p.sub_(lr_step)
                #print(f"post-step {p.data}")



            '''


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
            '''
