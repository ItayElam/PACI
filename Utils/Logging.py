import os
import sys
from torch.utils.tensorboard import SummaryWriter

class TeeOutput:
    def __init__(self, log_file):
        self.log_file = log_file
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log = open(self.log_file, 'a')

    def write(self, message):
        self.original_stdout.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.original_stdout.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def redirect_output_to_file(log_file):
    tee = TeeOutput(log_file)
    sys.stdout = tee
    sys.stderr = tee



class TensorBoardLogger:
    def __init__(self, log_dir, purge_step):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir, purge_step=purge_step)

    def log_metrics(self, metrics, step, prefix=""):
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def close(self):
        self.writer.flush()
        self.writer.close()