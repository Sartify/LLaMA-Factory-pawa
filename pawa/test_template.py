from llamafactory.data.template import TEMPLATES
from llamafactory.data.converter import SharegptDatasetConverter
from transformers import AutoTokenizer
import json


tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")

t = TEMPLATES["gemma3-pawa"]


def get_weather(location: str, unit: str) -> float:
    """
    Get the current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, Country"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])

    Returns:
        The current temperature at the specified location in the specified units, as a float.
    """
    return 25.0


tools = [get_weather]

message = [
    {
        "role": "system",
        "content": "You are a helpful assistant that can answer questions about the weather.",
    },
    {
        "role": "user",
        "content": "What is the weather in Paris, France?",
    },
]


t.fix_jinja_template(tokenizer=tokenizer)
# print(tokenizer.chat_template)

# print(t._get_jinja_template(tokenizer))
# print(tokenizer.chat_template)
#
#
msg = tokenizer.apply_chat_template(
    message,
    tools=tools,
    tokenize=False,
)
#
print(msg)
