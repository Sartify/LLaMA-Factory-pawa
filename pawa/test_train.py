from typing import Any

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.customized_utils import fix_chat_template_for_processor
from llamafactory.hparams import parser, read_args
from llamafactory.model import load_tokenizer

from llamafactory.hparams.data_args import DataArguments
from llamafactory.hparams.model_args import ModelArguments
from llamafactory.hparams.training_args import TrainingArguments


train_args = TrainingArguments(
    auto_output_dir=True,
    auto_output_root="outputs/test_outputs",
    report_to="none",
)
model_args = ModelArguments(model_name_or_path="google/gemma-3-4b-it")
data_args = DataArguments(
    dataset="xlam-function-calling-60k-sharegpt",
    template="gemma3-pawa",
    preprocessing_num_workers=8,
    max_samples=10,
    # overwrite_cache=False,
    disable_caching=True,
)

# run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)

tokenizer_module = load_tokenizer(model_args)
tokenizer = tokenizer_module["tokenizer"]
template = get_template_and_fix_tokenizer(tokenizer, data_args)
dataset_module = get_dataset(template, model_args, data_args, train_args, stage="sft", **tokenizer_module)
# print(dataset_module)
# print(tokenizer_module)
# print(tokenizer_module["processor"].tokenizer.chat_template)
fix_chat_template_for_processor(tokenizer_module["processor"])
print(tokenizer_module["processor"].chat_template)
tokenizer_module["processor"].save_pretrained("outputs/test_outputs")
# for key in tokenizer_module:
#     print()
for data in dataset_module["train_dataset"]:
    text = tokenizer.decode(data["input_ids"], skip_special_tokens=False)
    print(text)
    print(data)
