import torch
import threading
import torch.multiprocessing as mp
import copy

from Models.base import BaseModel
from .ModelConfig import TrainingConfig, LayerConfig
from .Communications import P2PQueue
from .Workers import LayerWorker
from .Constants import Signals
from .ModelPartitioner import GetPartitions

import time
import os
import torch.distributed.rpc as rpc
import warnings
import traceback
import signal

warnings.filterwarnings("ignore", category=UserWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)


class PACIException(Exception):
    def __init__(self, message):
        self.message = message


class SharedResults:
    def __init__(self):
        self._shared_dict = {}
        self._ready_flag = False

    def set_data(self, data):
        self._shared_dict.update(data)
        self._ready_flag = True

    def is_ready(self):
        return self._ready_flag

    def wait_until_ready(self, wait_time=0.1):
        while not self._ready_flag:
            if wait_time:
                time.sleep(wait_time)

    def get_data(self, wait=False):
        if wait:
            self.wait_until_ready()
        if not self.is_ready():
            raise PACIException("Data is not ready yet!")
        return self._shared_dict

def split_to_microbatchs(tensor, n, dim=0):
    total = tensor.size(dim)
    base = total // n
    extra = total % n

    sizes = [base + 1 if i < extra else base for i in range(n)]
    return list(torch.split(tensor, sizes, dim=dim))


class PACIModel:
    def __init__(self, model: BaseModel, config: TrainingConfig, measure_iterations=100):
        if os.environ.get("paci_suppress_output", "false") == "true":
            import sys
            sys.stdout = open(os.devnull, 'w')
        
        self.__config = config
        self.__model = model

        self.__ctx = mp.get_context('spawn')
        self.__manager = self.__ctx.Manager()
        self.__main_device = "cuda:0" if "cuda" in self.__config.device else "cpu"
        self.__dist_world_size = 1

        self.__partitioner = GetPartitions(self.__config)
        self.__model_state_dict = []
        self.__optimizers_state_dict = []
        self.__schedulers_state_dict = []
        self.__pending_results = []

        self.__split_indices = None
        self.__initialized = False
        self.__training = None

        self._measure_iterations = measure_iterations

        self.sent_without_recv = 0

        self.input_device = None
        self.target_device = None

    def load_state_dict(self, state_dict, input_sample):
        if not self.__initialized:
            self.__split_indices = state_dict['split_indices']
            self.__split_indices, self.__shapes = self.__partitioner.partition_model(self.__model, input_sample, state_dict['split_indices'], timing_repeat=self._measure_iterations, max_mem_usage=torch.cuda.mem_get_info()[1])
            self.__model_state_dict = (state_dict["model_state_dict"])
            self.__optimizers_state_dict = (state_dict["optimizers_state_dict"])
            self.__schedulers_state_dict = (state_dict.get("schedulers_state_dict", []))
            
        else:
            raise PACIException("Loading state dict must occur before model initialization")

    def init(self, input_sample, split_indices=None):
        if not self.__initialized:
            if self.__split_indices is None:
                self.__split_indices, self.__shapes = self.__partitioner.partition_model(self.__model, input_sample, indices=split_indices, timing_repeat=self._measure_iterations, max_mem_usage=torch.cuda.mem_get_info()[1])
                self.split_indices = self.__split_indices
            del self.__partitioner
            self.__dist_world_size = self.__config.split_to + 1

            self.__build_comm(self.__shapes)
            print("Main process: Building comm done!")
            self.__build_workers(self.__shapes)
            print("Main process: Building workers and comms done!")

            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29358'

            self.input_device = self.__devices[0]
            self.target_device = self.__devices[-1]

            first_rank = 1
            final_rank = self.__config.split_to
            mapping = {
                f"worker{first_rank}": {
                    self.__devices[0]: self.__devices[0]
                    },
                f"worker{final_rank}": {
                    self.__devices[-1]: self.__devices[-1]
                    },
            }

            rpc.init_rpc(f"worker{0}", rank=0, world_size=self.__dist_world_size,
                rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                    device_maps=mapping,
                )
            )
            
            self.__data_queue = P2PQueue(None, first_rank, 0, torch.device('cpu'), self.__devices[0], tag="data")
            self.__labels_queue = P2PQueue(None, final_rank, 0, torch.device('cpu'), self.__devices[-1], tag="labels")
            self.__results_queue = P2PQueue(final_rank, None, 0, torch.device('cpu'), None, tag="results")
            
            self.__layer_to_main_queue = [P2PQueue(i, None, 0, torch.device('cpu'), None, tag="l2m") for i in
                                            range(1, self.__config.split_to + 1)]
            self.__main_to_layer_queue = [P2PQueue(None, i, 0, torch.device('cpu'), self.__devices[i-1], tag="m2l") for i in
                                        range(1, self.__config.split_to + 1)]
            
            if len(self.__model_state_dict):
                self._load_model_states(self.__model_state_dict)
            if len(self.__optimizers_state_dict):
                self._load_optimizers_states(self.__optimizers_state_dict)
            if len(self.__schedulers_state_dict) and self.__config.lr_scheduler_class is not None:
                self._load_schedulers_states(self.__schedulers_state_dict)
            print("Main process: Loading model states done!")
            self.__keep_alive = True
            self.results_thread = threading.Thread(target=self._results_thread_worker, daemon=True)
            self.results_thread.start()
            self.__initialized = True
            self.train()
            torch.cuda.empty_cache()

            def _handle_sigint(signum, frame):
                try:
                    print("PACI Model: SIGINT received, stopping training...", flush=True)
                    self.stop_training()
                except Exception as e:
                    print(f"Error during stop_training: {e}", flush=True)
                finally:
                    os._exit(130)

            signal.signal(signal.SIGINT, _handle_sigint)

        else:
            raise PACIException("Model already initialized, call stop_training before reinitializing model")

    def _results_thread_worker(self):
        while self.__keep_alive:
            try:
                data_type, data, extra_data = self.__results_queue.get()
                self.__pending_results.pop(0).set_data(data)
                self.sent_without_recv -= 1
            except Exception as e:
                print(f"Main process: Error in results thread: {e}", flush=True)
                traceback.print_exc()

    def __call__(self, input_data: torch.Tensor, target: torch.Tensor, dont_wait=False) -> SharedResults:
        if self.__initialized:
            self.__check_reraise_exceptions()

            input_microbatches = split_to_microbatchs(input_data, self.__config.grad_accumulation)
            target_microbatches = split_to_microbatchs(target, self.__config.grad_accumulation)
            all_results = []
            for micro_in, micro_target in zip(input_microbatches, target_microbatches):
                while (not dont_wait) and (self.sent_without_recv >= (self.__config.split_to + 4)):
                    time.sleep(0.001)
                    self.__check_reraise_exceptions()
                self.__data_queue.put(micro_in)

                results_dict = SharedResults()
                
                self.__labels_queue.put(micro_target)
                self.__pending_results.append(results_dict)
                self.sent_without_recv += 1
                all_results.append(results_dict)
            return all_results
        else:
            raise PACIException("Initialize model before calling model")

    def __check_reraise_exceptions(self):
        for i, q in enumerate(self.__layer_to_main_queue):
            if not q.empty():
                data_type, data, extra_data = q.get()
                if data == Signals.exception:
                    raise Exception(f"Layer {i} got exception {extra_data}")
                else:
                    print(f"Layer {i} to main queue had unexpected data {data} | {extra_data}")
                    
    def to_torch_model(self):
        if self.__initialized:
            for i, state in enumerate(self._get_state_dict()):
                self.__model.layers[i].load_state_dict(state)
            model = copy.deepcopy(self.__model.cpu())
            all_layers = torch.nn.Sequential()
            for layer in model.layers:
                for layr in layer:
                    all_layers.append(layr)
            model.layers = all_layers
            return model
        else:
            raise PACIException("Initialize model before calling to_torch_model")

    def state_dict(self):
        if self.__initialized:
            state = {
                "model_state_dict": self._get_state_dict(),
                "optimizers_state_dict": self._get_optimizers_state_dict(),
                "schedulers_state_dict": self._get_schedulers_state_dict(),
                "split_indices": self.__split_indices,
                "block_io_shapes": self.__model.block_io_shapes
            }
            return state
        else:
            raise PACIException("Initialize model before calling state_dict")

    def train(self):
        if self.__initialized:
            if not self.__training:
                self.__training = True
                self.__send_and_wait(Signals.training_mode)
        else:
            raise PACIException("Initialize model before calling train")

    def eval(self):
        if self.__initialized:
            if self.__training:
                self.__training = False
                self.__send_and_wait(Signals.eval_mode)
        else:
            raise PACIException("Initialize model before calling eval")

    def half(self):
        if self.__initialized:
            self.__send_and_wait(Signals.model_half)
        else:
            raise PACIException("Initialize model before calling half")

    def float(self):
        if self.__initialized:
            self.__send_and_wait(Signals.model_float)
        else:
            raise PACIException("Initialize model before calling float")

    def step_scheduler(self):
        if self.__initialized:
            self.__send_and_wait(Signals.step_sched)
        else:
            raise PACIException("Initialize model before calling step_scheduler")

    def epoch_done(self):
        if self.__initialized:
            self.__send_and_wait(Signals.epoch_done)
        else:
            raise PACIException("Initialize model before calling stop_training")

    def stop_training(self):
        if self.__initialized:
            self.__initialized = False
            self.__keep_alive = False
            print("Main process: Stopping training...")
            self.__send_and_wait(Signals.stop)
            self.__processes = []
            self.__close_comm()
            print("Main process: About to shutdown rpc...")
            rpc.shutdown()
    
        else:
            raise PACIException("Initialize model before calling stop_training")

    def _get_state_dict(self):
        self.__send_and_wait(Signals.send_layers)
        model_states = []
        for i, q in enumerate(self.__layer_to_main_queue):
            data_type, data, extra_data = q.get()
            model_states.append(data)
        return model_states

    def _get_optimizers_state_dict(self):
        self.__send_and_wait(Signals.send_optimizers)
        optimizers_states = []
        for i, q in enumerate(self.__layer_to_main_queue):
            data_type, data, extra_data = q.get()
            optimizers_states.append(data)
        return optimizers_states

    def _get_schedulers_state_dict(self):
        self.__send_and_wait(Signals.send_schedulers)
        schedulers_states = []
        for i, q in enumerate(self.__layer_to_main_queue):
            data_type, data, extra_data = q.get()
            schedulers_states.append(data)
        return schedulers_states

    def _load_model_states(self, model_state_dicts):
        self.__send_and_wait(Signals.pause)
        for q, state_dict in zip(self.__main_to_layer_queue, model_state_dicts):
            q.put(state_dict)
        self.__send_and_wait(Signals.load_model_state_dict)

    def _load_optimizers_states(self, optimizers_state_dicts):
        self.__send_and_wait(Signals.pause)
        for q, state_dict in zip(self.__main_to_layer_queue, optimizers_state_dicts):
            q.put(state_dict)
        self.__send_and_wait(Signals.load_optimizer_state_dict)

    def _load_schedulers_states(self, schedulers_state_dicts):
        self.__send_and_wait(Signals.pause)
        for q, state_dict in zip(self.__main_to_layer_queue, schedulers_state_dicts):
            q.put(state_dict)
        self.__send_and_wait(Signals.load_scheduler_state_dict)

    def __send_and_wait(self, sig):
        self.__check_reraise_exceptions()
        if sig is not None:
            self.__data_queue.put(sig)
        for q in self.__layer_to_main_queue:
            q.get()  # acknowledgment signal

    def __build_comm(self, shapes):
        self.__devices = []

        gpu_plan = [i % (self.__config.split_to // self.__config.virtual_stages) for i in range(self.__config.split_to)]
        for i in range(self.__config.split_to):
            next_gpu = torch.device(f"cuda:{gpu_plan[i]}")
            self.__devices.append(next_gpu)
        print(f"Main process: Devices: {self.__devices}")
   
    def __close_comm(self):
        for q in [self.__data_queue, self.__labels_queue, self.__results_queue,
                  *self.__layer_to_main_queue, *self.__main_to_layer_queue]:
            if q is not None:
                q.close()

    def __build_workers(self, shapes):
        self.__processes = []
        worker_config = LayerConfig(self.__config)
        for idx in range(self.__config.split_to):
            self.__model.layers[idx].to('cpu')
            p = self.__ctx.Process(target=LayerWorker, args=(
                worker_config, self.__model.layers[idx], 
                self.__devices, idx))
            p.start()
            self.__processes.append(p)
        
