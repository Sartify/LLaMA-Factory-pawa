from llamafactory.hparams import get_infer_args, get_ray_args, get_train_args, read_args, parser
from llamafactory.train.sft import run_sft
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.model import load_tokenizer

from typing import Any, TYPE_CHECKING

args = read_args()
config = {"args": args, "callbacks": []}

args = config.get("args")
callbacks: list[Any] = config.get("callbacks")
# model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
model_args, data_args, training_args, finetuning_args, generating_args = parser._parse_train_args(args)

# run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)

tokenizer_module = load_tokenizer(model_args)
tokenizer = tokenizer_module["tokenizer"]
template = get_template_and_fix_tokenizer(tokenizer, data_args)
dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
print(dataset_module)
for data in dataset_module["train_dataset"]:
    text = tokenizer.decode(data["input_ids"], skip_special_tokens=False)
    print(text)
    print(data)
