from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy


h = "hello world"
p = "world hello"

bleu_score = sentence_bleu([list(h)], list(p), smoothing_function=SmoothingFunction().method3)
bleu_score_2 = sentence_bleu([h.split()], list(p.split()), smoothing_function=SmoothingFunction().method3)

print(bleu_score)
print(bleu_score_2)
