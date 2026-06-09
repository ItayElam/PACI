import os
import time
import copy
import queue
import signal
import traceback
import torch
import torch.distributed.rpc as rpc

from .Constants import Signals, DistTypes

from .Communications import P2PQueue
from .ModelConfig import LayerConfig, PipelineMethods, LRstepMode, PACIMode


def signal_handler(signum, frame):
    pass

default_handler = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, signal_handler)


class DemiOptimizer:
    def __init__(self, *args, **kwargs):
        pass

    def zero_grad(self):
        pass

    def step(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return



def build_optimizer(layer, config: LayerConfig):
    if next(layer.parameters(), None) is not None:
        return config.build_optimizer_func(layer, **config.optimizer_params)
    else:
        print('No parameters detected in this layer')
        return DemiOptimizer()


class LayerWorker:
    def __init__(self, config: LayerConfig, layer, 
                 device_list, layer_idx):
        if os.environ.get("paci_suppress_output", "false") == "true":
            import sys
            sys.stdout = open(os.devnull, 'w')
            
        if os.environ.get("paci_benchmark_accuracy", "false") == "true":
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            print("WARNING: PACI is benchmarking accuracy - lower speed")

        self.device = device_list[layer_idx]
        self.next_device = device_list[layer_idx + 1] if layer_idx + 1 < len(device_list) else None
        self.previous_device = device_list[layer_idx - 1] if layer_idx - 1 >= 0 else None

        if self.device != torch.device('cpu'):
            torch.cuda.set_device(self.device)
            print(f"Worker {layer_idx} using GPU {self.device}")

        self.config = config
        self.layer: torch.nn.Module = layer
        self.layer_count = config.split_to
        self.layer_idx = layer_idx

        self.optimizer = build_optimizer(self.layer, self.config)

        self.loss_func = config.loss_function
        if self.config.allowed_lagging is not None:
            self.allowed_lagging = config.allowed_lagging
        else:
            self.allowed_lagging = (config.split_to - self.layer_idx)

        print(f"Layer {self.layer_idx} allowed lagging {self.allowed_lagging}")

        if config.lr_scheduler_class:
            self.scheduler: torch.optim.lr_scheduler._LRScheduler = config.lr_scheduler_class(self.optimizer, **config.lr_scheduler_params)
        else:
            self.scheduler: torch.optim.lr_scheduler._LRScheduler = DemiOptimizer()

        self.pipeline_mode = config.pipeline_mode
        self.perf_mem_mode = config.paci_mode
        self.max_grad_accumulation = config.grad_accumulation
        self.current_grad_accumulation = 1
        self.inference_without_update = 0

        self.num_micro_batch_forward = 0
        self.num_micro_batch_backward = 0

        self.batch_idx = 0
        self.epochs_done = 1
        self.epoch_loss = 0
        self.eval_mode = False
        self.current_state = Signals.pause
        
        self.use_amp = False
        self.cought_exception = False

        self.setup()
        self.run()
        
    def setup(self):
        self.layer = self.layer.to(self.device)
        self.rank = self.layer_idx + 1
        
        print(f"Worker {self.layer_idx}: world_size={self.config.split_to+1}, rank={self.rank}")
        print(f"Worker {self.layer_idx}: About to initialize distributed group...")
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29358'

        mapping = {}
        if self.next_device is not None:
            # mapping[f"worker{self.rank+1}"] = {self.device: self.next_device}
            mapping[f"worker{self.rank+1}"] = {self.device: self.device}
        if self.previous_device is not None:
            # mapping[f"worker{self.rank-1}"] = {self.device: self.previous_device}
            mapping[f"worker{self.rank-1}"] = {self.device: self.device}

        # mapping[f"worker{0}"] = {self.device: torch.device('cpu')}
        mapping[f"worker{0}"] = {self.device: self.device}

        rpc.init_rpc(f"worker{self.rank}", rank=self.rank, world_size=self.config.split_to + 1,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                device_maps=mapping,
                )
        )

        print(f"Worker {self.layer_idx}: Distributed group initialized!")

        self.data_queue = P2PQueue(self.rank-1, self.rank+1, self.rank, self.device, self.next_device, tag="data")
        self.grad_queue = P2PQueue(self.rank+1, self.rank-1, self.rank, self.device, self.previous_device, tag="grad")

        if self.layer_idx == self.config.split_to - 1:
            self.labels_queue = P2PQueue(0, None, self.rank, self.device, None, tag="labels")
            self.results_queue = P2PQueue(None, 0, self.rank, self.device, torch.device('cpu'), tag="results")
        else:
            self.labels_queue = None
            self.results_queue = None

        self.layer_to_main_queue: P2PQueue = P2PQueue(None, 0, self.rank, self.device, torch.device('cpu'), tag="l2m")
        self.main_to_layer_queue: P2PQueue = P2PQueue(0, None, self.rank, self.device, None, tag="m2l")

        self.input_tensor_history = queue.Queue()
        self.output_tensor_history = queue.Queue()

        if self.config.on_before_optimizer_step:
            self.config.on_before_optimizer_step = self.config.on_before_optimizer_step()
        
    def run(self):
        print(f"Layer {self.layer_idx} has pid {os.getpid()}")
        try:
            if self.device != torch.device("cpu"):
                torch.cuda.reset_peak_memory_stats(self.device)
            if self.layer_idx == self.layer_count - 1:
                self.forward = self.forward_last_layer
                self.backward = self.backward_last_layer
                self.run_last_layer()
            else:
                self.forward = self.forward_other_layers
                self.backward = self.backward_other_layers
                self.run_other_layers()
        except KeyboardInterrupt:
            print(f"[{self.layer_idx}] KeyBoard Interrupt")
        except Exception as e:
            print(f"[{self.layer_idx}] Error: {e}")
            self.layer_to_main_queue.put(Signals.exception, str(e))
            traceback.print_exc()
        finally:
            self.cleanup()
            print(f"Layer {self.layer_idx} done", flush=True)

    def cleanup(self):
        del self.layer

        if self.layer_to_main_queue is not None:
            self.layer_to_main_queue.close()
        if self.main_to_layer_queue is not None:
            self.main_to_layer_queue.close()
        del self.input_tensor_history
        del self.output_tensor_history
        del self.optimizer
        del self.scheduler
        if self.device != torch.device("cpu"):
            torch.cuda.empty_cache()

        if self.labels_queue is not None:
            self.labels_queue.close()
        if self.results_queue is not None:
            self.results_queue.close()
        if self.data_queue is not None:
            self.data_queue.close()
        if self.grad_queue is not None:
            self.grad_queue.close()

        print(f"Worker {self.layer_idx}: About to shutdown rpc...")
        rpc.shutdown()
        print(f"Worker {self.layer_idx}: Process group destroyed!")

    def wait_for_signal(self):
        while self.data_queue.empty():
            time.sleep(0.5)

    def handle_input_or_signal(self, input_data):
        data_type, tensor, extra_data = input_data
        self.current_state = Signals.play
        if data_type == DistTypes.other:
            if self.cought_exception:
                print(f"[worker {self.layer_idx}] passing {tensor} forwards after exception")
            self.data_queue.put(tensor)
        elif not self.cought_exception:
            self.forward(tensor, extra_data)

            self.num_micro_batch_forward += 1
            self.inference_without_update += 1
            self.batch_idx += 1
        else:
            print(f"[worker {self.layer_idx}] skipped forward because of exception")

    def handle_grad_or_signal(self, grad_or_label):
        data_type, tensor, extra_data = grad_or_label
        self.current_state = Signals.play

        if data_type == DistTypes.other:
            if self.cought_exception:
                print(f"[worker {self.layer_idx}] passing {tensor} backwards after exception")
            if self.layer_idx > 0:
                self.grad_queue.put(tensor)
            self.current_state = tensor
            self.handle_signals()
        elif not self.cought_exception:
            self.backward(tensor, extra_data)
            
            self.num_micro_batch_backward += 1
            self.inference_without_update -= 1
        else:
            print(f"[worker {self.layer_idx}] skipped backward because of exception")


    def _do_forward(self, input_tensor, extra_data):
        if (self.layer_idx > 0) and not self.eval_mode:
            input_tensor.requires_grad_(True)
            input_tensor.retain_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            if extra_data is not None: 
                output_tensor = self.layer((input_tensor, *extra_data))
            else:
                output_tensor = self.layer(input_tensor)
        return output_tensor

    def _do_opt_step(self):
        if self.current_grad_accumulation >= self.max_grad_accumulation:
            if self.config.on_before_optimizer_step:
                self.config.on_before_optimizer_step(self.layer)

            if self.perf_mem_mode == PACIMode.mem:
                self.optimizer.step()
            else:
                with torch.set_freeze_version_update():
                    self.optimizer.step()
            if self.config.lr_step_on == LRstepMode.step:
                self.scheduler.step()
            self.optimizer.zero_grad()
            self.current_grad_accumulation = 0
        self.current_grad_accumulation += 1

    def run_last_layer(self):
        while self.current_state != Signals.stop:
            try:
                if self.current_state == Signals.pause:
                    self.wait_for_signal()
                
                data_type, input_data, extra_data = self.data_queue.get()
                if data_type == DistTypes.other:
                    self.handle_grad_or_signal((data_type, input_data, extra_data))
                else:
                    self.handle_input_or_signal((data_type, input_data, extra_data))
                    data_type, target, extra_data = self.labels_queue.get()
                    self.handle_grad_or_signal((data_type, target, extra_data))
            except Exception as e:
                print(f"[{self.layer_idx}] Error: {e}")
                self.layer_to_main_queue.put(Signals.exception, str(e))
                traceback.print_exc()

                self.grad_queue.put(Signals.exception)
                self.cought_exception = True
    
        print(f"Layer {self.layer_idx}: Exited for signal {self.current_state}")

    def forward_last_layer(self, input_tensor, extra_data):
        output_tensor = self._do_forward(input_tensor, extra_data)            
        self.input_tensor_history.put(input_tensor)
        self.output_tensor_history.put(output_tensor)

    def backward_last_layer(self, target, _):
        input_tensor = self.input_tensor_history.get()
        output = self.output_tensor_history.get()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            loss = self.loss_func(output, target)
            orig_loss = loss.item()                
            if torch.isnan(loss):
                print(f"Got NAN loss: {loss}")

        if self.config.on_loss_computation:
            loss = self.config.on_loss_computation(loss, output, target)
            self.epoch_loss += loss.item()

        if not self.eval_mode:
            scaled_loss = loss / self.max_grad_accumulation
            scaled_loss.backward()

            if self.layer_idx > 0:
                grad = input_tensor.grad.detach()
                self.grad_queue.put(grad)

            self._do_opt_step()

        if self.config.output_verbose:
            self.results_queue.put({"validating": self.eval_mode, "batch_idx": self.batch_idx, "loss": orig_loss,
                "output": output, "target": target})          
        else:
            valid_tokens = int((target != self.config.ignore_token).sum().item())
            self.results_queue.put({"loss": float(orig_loss), "tokens": valid_tokens, "lr": self.scheduler.get_last_lr() if not isinstance(self.scheduler, DemiOptimizer) else self.config.optimizer_params['lr']})

    def run_other_layers(self):
        while self.current_state != Signals.stop:
            try:
                if self.current_state == Signals.pause:
                    self.wait_for_signal()
                while not self.grad_queue.empty():
                    grad_from_next_layer = self.grad_queue.get()
                    self.handle_grad_or_signal(grad_from_next_layer)
                
                forward_allowed = self.eval_mode | self.cought_exception
                if self.pipeline_mode == PipelineMethods.paci:
                    forward_allowed |= self.inference_without_update < self.allowed_lagging
                elif self.pipeline_mode == PipelineMethods.flush: 
                    forward_allowed |= self.inference_without_update < self.allowed_lagging and (self.layer_idx != 0 or \
                        (
                        (self.num_micro_batch_forward - self.num_micro_batch_backward) < self.allowed_lagging and \
                        (self.num_micro_batch_forward % self.max_grad_accumulation != 0 or (self.num_micro_batch_forward == self.num_micro_batch_backward))
                        ))
                else:
                    raise Exception(f"Invalid pipeline method {self.pipeline_mode}")
                if not self.data_queue.empty() and forward_allowed:
                    input_data = self.data_queue.get()
                    self.handle_input_or_signal(input_data)
            except Exception as e:
                print(f"[Worker {self.layer_idx}] Error: {e}")
                self.layer_to_main_queue.put(Signals.exception, str(e))
                self.data_queue.put(Signals.exception)
                self.cought_exception = True

        print(f"Layer {self.layer_idx}: Exited for signal {self.current_state}")

    def forward_other_layers(self, input_tensor, extra_data):

        if self.perf_mem_mode == PACIMode.mem:
            torch.set_grad_enabled(False)
        output_tensor = self._do_forward(input_tensor, extra_data)

        if self.perf_mem_mode == PACIMode.mem:
            torch.set_grad_enabled(not self.eval_mode)
        
        if isinstance(output_tensor, torch.Tensor):
            new_extra_data = None
        else:
            output_tensor, new_extra_data = output_tensor[0], output_tensor[1:]

        self.data_queue.put(output_tensor, new_extra_data)

        if self.eval_mode:
            return
        
        self.input_tensor_history.put({"input_tensor": input_tensor, "extra_data": extra_data})
        if self.perf_mem_mode == PACIMode.mem:
            self.output_tensor_history.put(None)
        else:
            self.output_tensor_history.put(output_tensor)

    def backward_other_layers(self, grad_from_next_layer, _):

        input_data = self.input_tensor_history.get()
        input_tensor = input_data['input_tensor']
        extra_data = input_data['extra_data']
            
        output_tensor: torch.Tensor = self.output_tensor_history.get()
        if output_tensor is None:
            output_tensor = self._do_forward(input_tensor, extra_data)
            if not isinstance(output_tensor, torch.Tensor):
                output_tensor = output_tensor[0]

        output_tensor.backward(grad_from_next_layer)

        if self.layer_idx > 0:
            grad = input_tensor.grad.detach()
            self.grad_queue.put(grad)

        self._do_opt_step()

    def handle_signals(self):
        if not isinstance(self.current_state, Signals.signal_type):
            return
        self.layer_to_main_queue.put(Signals.ack)
        torch.cuda.synchronize()

        if self.current_state == Signals.stop:
            exit(0)
        if self.current_state == Signals.step_sched:
            self.scheduler.step()
        if self.current_state == Signals.epoch_done:
            self.epochs_done += 1
            self.epoch_loss = 0
            self.inference_without_update = 0
            self.num_micro_batch_forward = 0
            self.num_micro_batch_backward = 0
        if self.current_state == Signals.training_mode:
            self.batch_idx = 0
            if os.environ.get("paci_benchmark_accuracy", "false") == "true":
                self.layer.eval()
                print("WARNING: PACI is benchmarking accuracy - training in eval mode")
            else:
                self.layer.train()
            torch.cuda.empty_cache()
            self.eval_mode = False
            torch.set_grad_enabled(True)
            self.input_tensor_history = queue.Queue()
            self.output_tensor_history = queue.Queue()
            self.inference_without_update = 0
            self.num_micro_batch_forward = 0
            self.num_micro_batch_backward = 0
        if self.current_state == Signals.eval_mode:
            self.batch_idx = 0
            self.layer.eval()
            self.eval_mode = True
            torch.set_grad_enabled(False)
            torch.cuda.empty_cache()
            self.input_tensor_history = queue.Queue()
            self.output_tensor_history = queue.Queue()
        if self.current_state == Signals.model_half:
            self.use_amp = True
        if self.current_state == Signals.model_float:
            self.use_amp = False
        if self.current_state == Signals.send_layers:
            self.layer_to_main_queue.put(copy.deepcopy(self.layer.state_dict()))
        if self.current_state == Signals.send_optimizers:
            self.layer_to_main_queue.put(copy.deepcopy(self.optimizer.state_dict()))
        if self.current_state == Signals.send_schedulers:
            self.layer_to_main_queue.put(copy.deepcopy(self.scheduler.state_dict()))
        if self.current_state == Signals.load_model_state_dict:
            state_dict = self.main_to_layer_queue.get()[1]
            self.layer.load_state_dict(state_dict, strict=True)
            self.layer.to(self.device)
        if self.current_state == Signals.load_optimizer_state_dict:
            state_dict = self.main_to_layer_queue.get()[1]
            self.optimizer = build_optimizer(self.layer, self.config)
            self.optimizer.load_state_dict(state_dict)
        if self.current_state == Signals.load_scheduler_state_dict:
            state_dict = self.main_to_layer_queue.get()[1]
            if self.config.lr_scheduler_class:
                self.scheduler: torch.optim.lr_scheduler._LRScheduler = self.config.lr_scheduler_class(self.optimizer, **self.config.lr_scheduler_params)
            else:
                self.scheduler: torch.optim.lr_scheduler._LRScheduler = DemiOptimizer()

            self.scheduler.load_state_dict(state_dict)
        if self.current_state == Signals.exception:
            self.cought_exception = True
            print(f"[Worker {self.layer_idx}] set cought exception to True")
        self.current_state = Signals.pause
        torch.cuda.synchronize()
