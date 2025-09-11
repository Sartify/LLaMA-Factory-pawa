import re
import os
from transformers import TrainerCallback
from transformers import TrainingArguments, TrainerControl, TrainerState
import sys
import torch
import gc
from llamafactory.extras.logging import get_logger
import json

logger = get_logger(__name__)


sys.path.append(os.getcwd())


class OnSaveEvaluationCallback(TrainerCallback):
    """A callback that performs evaluation when the model is saved."""

    def __init__(self, base_model_name: str | None, eval_batch_size: int = 8):
        super().__init__()
        if base_model_name is None:
            raise ValueError("base_model_name must be specified when using OnSaveEvaluationCallback.")
        self.base_model_name = base_model_name
        self.eval_batch_size = eval_batch_size

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Event called after a checkpoint save."""

        logger.info("Releasing GPU memory before evaluation...")
        torch.cuda.empty_cache()
        gc.collect()

        if state.is_local_process_zero:
            print("Saving model and clearing GPU memory...")
            gc.collect()
            torch.cuda.empty_cache()

        if state.is_local_process_zero:
            args.output_dir
            steps, checkpoint_dir = self.parse_checkpoint_dir(args.output_dir)

            assert steps == state.global_step, "the checkpoint step should be the same as the global step"

            evauluation_working_dir = os.path.abspath(os.path.join(args.output_dir, f"evaluation-{steps}"))
            cache_dir = os.path.abspath(os.path.join(evauluation_working_dir, "cache"))

            convert_output_dir = os.path.abspath(os.path.join(cache_dir, "converted_model"))
            torch_save_checkpoint_dir = os.path.abspath(os.path.join(cache_dir, "torch_save_checkpoint.pth"))

            os.makedirs(evauluation_working_dir, exist_ok=True)
            os.makedirs(cache_dir, exist_ok=True)
            os.makedirs(convert_output_dir, exist_ok=True)
            logger.info_rank0(f"Converting checkpoint {checkpoint_dir} to {convert_output_dir}")

            # convert models distributed
            os.system(
                f"python3 -m pawa.scripts.convert_ckpt_to_hf_model "
                f"--model_name {self.base_model_name} "
                f"--llama_factory_checkpoint_dir {os.path.join(args.output_dir, checkpoint_dir)} "
                f"--torch_save_checkpoint_dir {torch_save_checkpoint_dir} "
                f"--output_dir {convert_output_dir} "
                f"--template_name gemma3-pawa"
            )
            logger.info_rank0(f"Evaluating checkpoint {checkpoint_dir} in {evauluation_working_dir}")

            logger.info_rank0(f"Evaluating model in {convert_output_dir}")

            ret = os.system(
                f"CUDA_VISIBLE_DEVICES=0,1 python -m lm_eval --model hf "
                f"--model_args pretrained={convert_output_dir},device_map=auto "
                f"--tasks afrimmlu_direct_zul_prompt_1,afrimmlu_translate_zul_prompt_1 "
                f"--batch_size {self.eval_batch_size} "
                f"--output_path {os.path.join(evauluation_working_dir, 'results.json')} "
            )
            if ret != 0:
                raise RuntimeError("Evaluation failed.")

            # delete cache
            os.system(f"rm -rf {cache_dir}")

            # find result json
            result_json = self.find_ressult_json(evauluation_working_dir)

            with open(os.path.join(evauluation_working_dir, result_json)) as f:
                result = json.load(f)
                self.log_eval_results(self, kwargs["trainer"], result)

            logger.info_rank0(f"Evaluation results are saved in {result_json}")

    @staticmethod
    def log_eval_results(self, trainer, result: dict):
        trainer.log(
            dict(
                afrimmlu_direct_zul_prompt_1_accuracy=result["afrimmlu_direct_zul_prompt_1"]["accuracy"],
                afrimmlu_translate_zul_prompt_1_accuracy=result["afrimmlu_translate_zul_prompt_1"]["accuracy"],
            )
        )

    @staticmethod
    def find_ressult_json(evaluation_working_dir: str) -> str:
        candidates = os.listdir(evaluation_working_dir)
        pattern = re.compile(r"^results_.*\.json$")
        for candidate in candidates:
            match = pattern.match(candidate)
            if match:
                return os.path.join(evaluation_working_dir, candidate)

        raise ValueError(f"No results_*.json found in {evaluation_working_dir}")

    @staticmethod
    def parse_checkpoint_dir(output_dir: str) -> int:
        """checkpoint-14000"""
        candidates = os.listdir(output_dir)
        pattern = re.compile(r"^checkpoint-(\d+)$")

        checkpoints_dir = []
        for candidate in candidates:
            match = pattern.match(candidate)
            if match:
                checkpoints_dir.append((int(match.group(1)), candidate))  # (14000, "checkpoint-14000")

        checkpoints_dir = sorted(checkpoints_dir, key=lambda x: x[0], reverse=True)
        if len(checkpoints_dir) == 0:
            raise ValueError(f"No checkpoint found in {output_dir}")
        return checkpoints_dir[0]  # return the latest checkpoint number
