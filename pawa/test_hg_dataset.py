from datasets import Dataset


# 示例数据集
dataset = Dataset.from_dict({"sentence": ["This is. a sentence.", "Another. example sentence."]})


# 定义映射函数
def split_sentence(batch):
    sentences = batch["sentence"]
    # 将每个句子拆分为多个子句
    split_sentences = [sub_sentence for sentence in sentences for sub_sentence in sentence.split(".")]
    return {"new_sentence": split_sentences}


# 使用 map 方法扩展数据集
expanded_dataset = dataset.map(split_sentence, batched=True, remove_columns=["sentence"])

# 查看结果
print(expanded_dataset["new_sentence"])
