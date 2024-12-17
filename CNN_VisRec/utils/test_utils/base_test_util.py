import os
import torch.distributed as dist

from utils import dist_util, logger
from utils.debug_util import *
from utils.data_utils.transform_util import *


class TestLoop:
    def __init__(
            self,
            *,
            model,
            data,
            batch_size,
            log_interval,
            model_save_dir="",
            resume_checkpoint="",
            output_dir="",
            use_fp16=False,
            debug_mode=False,
    ):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.model_save_dir = model_save_dir
        if self.model:
            assert resume_checkpoint != "", "the model for test must be specified."
        else:
            assert resume_checkpoint == "", "do not use any model."
        self.resume_checkpoint = resume_checkpoint
        self.output_dir = output_dir
        self.use_fp16 = use_fp16
        self.debug_mode = debug_mode

        self.step = 0

        if self.model:
            self._load_parameters()
            logger.log(f"This model contains {count_parameters_in_M(self.model)}M parameters")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    # ytxie: We use the simplest method to load model parameters.
    def _load_parameters(self):
        model_checkpoint = os.path.join(self.model_save_dir, self.resume_checkpoint)
        logger.log(f"loading model from checkpoint: {model_checkpoint}...")
        self.model.load_state_dict(
            th.load(
                model_checkpoint,
                map_location="cpu",
            )
        )
        self.model.to(dist_util.dev())
        if self.use_fp16:
            self.model.convert_to_fp16()

        self.model.eval()

    # ytxie: This function wraps the whole test process.
    def run_loop(self):
        for data_item in self.data:
            self.forward_backward(data_item)
            self.step += 1
            if self.debug_mode or self.step % self.log_interval == 0:
                show_gpu_usage(f"step: {self.step}, device: {dist.get_rank()}", idx=dist.get_rank())

            if self.step % self.log_interval == 0:
                logger.log(f"have test {self.step} steps")

        dist.barrier()





