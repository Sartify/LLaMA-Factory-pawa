from transformers import PreTrainedTokenizer
from transformers.processing_utils import ProcessorMixin

from . import logging

logger = logging.get_logger(__name__)


def fix_chat_template_for_processor(processor: ProcessorMixin):
    """Fixes the tokenizer for the processor by setting the tokenizer attribute."""
    assert hasattr(processor, "chat_template"), (
        "Processor does not have a chat_template attribute. "
        "This function is intended for processors that support chat templates."
    )
    assert hasattr(processor, "tokenizer"), (
        "Processor does not have a tokenizer attribute. "
        "This function is intended for processors that support tokenizers."
    )
    try:
        processor.chat_template = processor.tokenizer.chat_template
        logger.info_rank0("Successfully set chat_template in processor.")
    except ValueError as e:
        logger.info_rank0(
            f"Failed to set chat_template in processor: {e}. This may not be supported by the processor class."
        )
