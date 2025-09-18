from datasets import load_dataset
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer


dataset = load_dataset("sartifyllc/swahili-pretrained-dataset")
dataset = dataset["swahili_translation"]

print(dataset.features)

for data in dataset:
    print(data["text"])
    break
