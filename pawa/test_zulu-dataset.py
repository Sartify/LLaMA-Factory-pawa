from datasets import load_dataset
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer


BATCH_SIZE_PER_DEVICE = 12
NUM_DEVICES = 2


dataset = load_dataset("sartifyllc/aya-zulu-pretrained")
dataset_len = len(dataset["train"])


print(dataset_len / (BATCH_SIZE_PER_DEVICE * NUM_DEVICES))
