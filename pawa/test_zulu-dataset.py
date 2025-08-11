from datasets import load_dataset
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer

dataset = load_dataset("sartifyllc/aya-zulu-pretrained")
print(len(dataset["train"]))
print(dataset["train"].features)
print(dataset["train"][0]["text"])
