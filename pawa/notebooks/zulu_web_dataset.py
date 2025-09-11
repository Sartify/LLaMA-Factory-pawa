import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import datasets
    import matplotlib.pyplot as plt
    from datasets import load_dataset, Dataset


@app.cell
def _():
    train_dataset = load_dataset("HuggingFaceFW/fineweb-2", name="zul_Latn", split="train")
    test_dataset = load_dataset("HuggingFaceFW/fineweb-2", name="zul_Latn", split="test")
    return test_dataset, train_dataset


@app.cell
def _(mo):
    mo.md(r"""## Zulu dataset Analysis""")
    return


@app.cell
def _(test_dataset, train_dataset):
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Columns in the dataset: {train_dataset.column_names}")
    print(f"Example from train dataset: {train_dataset[0]}")
    print(f"Example from test dataset: {test_dataset[0]}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Zulu dataset figure
    $$
    \alpha + \beta
    $$
    """
    )
    return


@app.cell
def _(train_dataset):
    # Plot the distribution of text lengths in the training dataset
    text_lengths = [len(example["text"]) for example in train_dataset]
    plt.hist(text_lengths, bins=50, color="blue", alpha=0.7)
    plt.title("Distribution of Text Lengths in Training Dataset")
    plt.xlabel("Text Length")
    plt.ylabel("Frequency")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
