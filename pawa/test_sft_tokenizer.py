from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("outputs/gemma3_test/checkpoint-500", local_files_only=True)

print(tokenizer.chat_template)
