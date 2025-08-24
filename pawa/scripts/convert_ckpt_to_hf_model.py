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


model_name = "google/gemma-3-270m"
template_name = "gemma3-pawa"

LLAMA_FACTORY_CHECKPOINT_DIR = "outputs/pawa-270m-zulu-pt-freeze/2025-08-21_22-15-49/checkpoint-5992"
TORCH_SAVE_CHECKPOINT_DIR = "outputs/torch_save_checkpoint.pth"
OUTPUT_DIR = "outputs/test_output"

if __name__ == "__main__":
    template = TEMPLATES[template_name]

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
    # save tokenizer and tempaltes
    processor = AutoProcessor.from_pretrained(model_name)

    if isinstance(processor, PreTrainedTokenizer) or isinstance(processor, PreTrainedTokenizerFast):
        template.fix_jinja_template(processor)
    else:
        template.fix_jinja_template(processor.tokenizer)

    model.save_pretrained(OUTPUT_DIR, max_shard_size="2GB")
    processor.save_pretrained(OUTPUT_DIR)
