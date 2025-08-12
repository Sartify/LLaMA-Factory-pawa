import os

import torch
import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save, torch_save_to_dcp

from transformers import (
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
)

model_name = "google/gemma-3-4b-it"
LLAMA_FACTORY_CHECKPOINT_DIR = "outputs/pawa-zulu-pt/checkpoint-3500"
TORCH_SAVE_CHECKPOINT_DIR = "outputs/torch_save_checkpoint.pth"
OUTPUT_DIR = "outputs/test_output"


if __name__ == "__main__":
    dcp_ckpt_path = os.path.join(LLAMA_FACTORY_CHECKPOINT_DIR, "pytorch_model_fsdp_0")
    dcp_to_torch_save(dcp_ckpt_path, TORCH_SAVE_CHECKPOINT_DIR)
    #
    ckpt = torch.load(TORCH_SAVE_CHECKPOINT_DIR, map_location="cpu")
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(
        ckpt["model"],
        strict=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_FACTORY_CHECKPOINT_DIR)
    processor = AutoProcessor.from_pretrained(LLAMA_FACTORY_CHECKPOINT_DIR)

    model.save_pretrained(OUTPUT_DIR, max_shard_size="2GB")
    tokenizer.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
