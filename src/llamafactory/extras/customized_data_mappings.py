import json
import numpy


class ShuffleTools:
    def __init__(self, p=0.8):
        self.p = p

    def __call__(self, example):
        if numpy.random.rand() < self.p:
            # shuffle the tools
            tools_obj = json.loads(example["tools"])
            assert isinstance(tools_obj, list), "tools should be a list of objects"
            numpy.random.shuffle(tools_obj)
            example["tools"] = json.dumps(tools_obj)
        return example


def build_xlam_function_calling_data_mapping():
    """Build the customized data mapping for xlam function calling."""
    shuffle_tools = ShuffleTools(0.8)

    def mapping(example):
        example = shuffle_tools(example)
        return example

    return mapping


CUSTOMIZED_DATA_MAPPING = {
    "Beryex/xlam-function-calling-60k-sharegpt": build_xlam_function_calling_data_mapping(),
}
