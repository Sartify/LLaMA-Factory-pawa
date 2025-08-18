from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")

with open("pawa/jinjas/gemma3-pawa-nomm-chat-template.jinja", "w") as f:
    f.write(tokenizer.chat_template)
