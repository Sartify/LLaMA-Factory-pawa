import os
from re import TEMPLATE

import torch
import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import (
    AutoConfig,
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from llamafactory.data.template import TEMPLATES
from argparse import ArgumentParser


# MODEL_NAME = "google/gemma-3-270m"
MODEL_NAME = "google/gemma-3-1b-pt"
TEMPLATE_NAME = "gemma3-pawa"

LLAMA_FACTORY_CHECKPOINT_DIR = "outputs/pawa-1b-zulu-pt-freeze/2025-08-28_01-16-41/checkpoint-15000"
TORCH_SAVE_CHECKPOINT_DIR = "outputs/torch_save_checkpoint.pth"
OUTPUT_DIR = "outputs/test_output"


def main(args):
    template = TEMPLATES[args.template_name]

    dcp_ckpt_path = os.path.join(args.llama_factory_checkpoint_dir, "pytorch_model_fsdp_0")
    dcp_to_torch_save(dcp_ckpt_path, args.torch_save_checkpoint_dir)
    #
    ckpt = torch.load(args.torch_save_checkpoint_dir, map_location="cpu")
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(
        ckpt["model"],
        strict=True,
    )
    # save tokenizer and tempaltes
    processor = AutoProcessor.from_pretrained(args.model_name)

    if isinstance(processor, PreTrainedTokenizer) or isinstance(processor, PreTrainedTokenizerFast):
        template.fix_jinja_template(processor)
    else:
        template.fix_jinja_template(processor.tokenizer)

    model.save_pretrained(OUTPUT_DIR, max_shard_size="2GB")
    processor.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--template_name", type=str, default=TEMPLATE_NAME)
    parser.add_argument("--llama_factory_checkpoint_dir", type=str, default=LLAMA_FACTORY_CHECKPOINT_DIR)
    parser.add_argument("--torch_save_checkpoint_dir", type=str, default=TORCH_SAVE_CHECKPOINT_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    main(parser.parse_args())
