from typing import Callable, Optional, Any, Union, Type

import torch
from torch.optim.lr_scheduler import LRScheduler

class PipelineMethods:
    paci = "paci"
    flush = "flush"

class PACIMode:
    mem = "mem"
    perf = "perf"

class LRstepMode:
    step = "step"
    custom = "custom"

class TrainingConfig:
    """
    Extended configuration class for managing the training loop.
    """

    def __init__(
            self,
            model_class: torch.nn.Module,

            # Training Parameters
            build_optimizer_func: Type[Callable],
            optimizer_params: dict,
            total_steps: int,
            loss_function: Callable[..., torch.Tensor],
            log_training_metrics_every: int = 10000000000000000,
            checkpoint_every: int = 10000000000000000,
            evaluate_every: int = 10000000000000000,
            grad_accumulation: int = 1,
            allowed_lagging: Optional[int] = None,
            # PACI mode
            pipeline_mode: PipelineMethods = PipelineMethods.paci,
            paci_mode: PACIMode = PACIMode.perf,

            # Backend
            backend="",
            init_method="tcp://localhost:12355",

            # Model Parameters
            split_to=1,
            virtual_stages=1,

            # Scheduler Parameters
            lr_scheduler_class: Optional[Type[LRScheduler]] = None,
            lr_scheduler_params: Optional[dict] = None,
            lr_step_on: LRstepMode = LRstepMode.custom,

            # Device Parameters
            device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",

            # Hooks
            on_before_optimizer_step: Optional[Callable[[Any, Any], None]] = None,
            on_loss_computation: Optional[Callable[[torch.Tensor, Any, Any], torch.Tensor]] = None,

            ignore_token=None,
            output_verbose=True,
    ):
        # Model Parameters
        self.model_class = model_class
        self.split_to = split_to
        self.virtual_stages = virtual_stages

        # PACI params
        self.pipeline_mode = pipeline_mode
        self.paci_mode = paci_mode

        # Backend
        self.backend = backend
        self.init_method = init_method

        # Training Parameters
        self.build_optimizer_func = build_optimizer_func
        self.optimizer_params = optimizer_params
        self.total_steps = total_steps
        self.loss_function = loss_function
        self.grad_accumulation = grad_accumulation
        self.allowed_lagging = allowed_lagging
        self.log_training_metrics_every = log_training_metrics_every
        self.checkpoint_every = checkpoint_every
        self.evaluate_every = evaluate_every

        # Scheduler Parameters
        self.lr_scheduler_class = lr_scheduler_class
        self.lr_scheduler_params = lr_scheduler_params
        self.lr_step_on = lr_step_on
        # Device Parameters
        self.device = device

        # Hooks
        self.on_before_optimizer_step = on_before_optimizer_step
        self.on_loss_computation = on_loss_computation

        self.ignore_token = ignore_token
        self.output_verbose=output_verbose

    def __repr__(self):
        return (
            f"TrainingConfig(\n"
            f"  model_class={self.model_class},\n"
            f"  pipeline_method={self.pipeline_mode},\n"
            f"  split_to={self.split_to},\n"
            f"  grad_accumulation={self.grad_accumulation}\n"
            f"  allowed_lagging={self.allowed_lagging}\n"
            f"  optimizer={(self.build_optimizer_func).__name__},\n"
            f"  optimizer_params={self.optimizer_params},\n"
            f"  total_steps={self.total_steps},\n"
            f"  loss_function={self.loss_function.__class__.__name__},\n"
            f"  lr_scheduler={(self.lr_scheduler_class).__name__ if self.lr_scheduler_class else None},\n"
            f"  scheduler_params={self.lr_scheduler_params},\n"
            f"  lr_step_on={self.lr_step_on},\n"
            f"  device={self.device},\n"
            f"  hooks_defined={[hook for hook in dir(self) if hook.startswith('on_') and getattr(self, hook)]}\n"
            f")"
        )


class LayerConfig:
    def __init__(self, config: TrainingConfig):
        self.split_to = config.split_to
        self.allowed_lagging = config.allowed_lagging
        self.loss_function = config.loss_function
        self.lr_scheduler_class = config.lr_scheduler_class
        self.lr_scheduler_params = config.lr_scheduler_params
        self.lr_step_on = config.lr_step_on
        self.paci_mode = config.paci_mode
        self.pipeline_mode: PipelineMethods = config.pipeline_mode
        self.grad_accumulation = config.grad_accumulation
        self.build_optimizer_func = config.build_optimizer_func
        self.optimizer_params = config.optimizer_params
        self.on_loss_computation = config.on_loss_computation
        self.on_before_optimizer_step = config.on_before_optimizer_step
        self.backend = config.backend
        self.init_method = config.init_method
        self.output_verbose = config.output_verbose
        self.ignore_token = config.ignore_token
