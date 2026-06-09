


class Sig:
    _num_name_map = {
    0: "play",
    1: "pause",
    2: "stop",
    3: "send_layers",
    4: "epoch_done",
    5: "step_sched",
    6: "training_mode",
    7: "eval_mode",
    8: "model_half",
    9: "model_float",
    10: "send_optimizers",
    11: "send_schedulers",
    12: "load_optimizer_state_dict",
    13: "load_scheduler_state_dict",
    14: "load_model_state_dict",
    15: "exception",
    50: "queue_stop",
    99: "ack",
}
    def __init__(self, num):
        self.num = num

    def __eq__(self, other):
        if isinstance(other, Sig):
            return self.num == other.num
        return False

    def __repr__(self):
        return f"{type(self).__name__}({self._num_name_map[self.num]})"


class Signals:
    signal_type = Sig
    # External Signals
    play = Sig(0)
    pause = Sig(1)
    stop = Sig(2)
    send_layers = Sig(3)
    epoch_done = Sig(4)
    step_sched = Sig(5)
    training_mode = Sig(6)
    eval_mode = Sig(7)
    model_half = Sig(8)
    model_float = Sig(9)
    send_optimizers = Sig(10)
    send_schedulers = Sig(11)
    load_optimizer_state_dict = Sig(12)
    load_scheduler_state_dict = Sig(13)
    load_model_state_dict = Sig(14)
    exception = Sig(15)

    # Internal signals
    queue_stop = Sig(50)

    ack = Sig(99)

class DistTypes:
    tensor = 0
    other = 1