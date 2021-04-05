from utils import STOPWORDS, LexRank
from path import Path
from myRouge import rouge_1
import rouge
import nltk.data

texts = []
text_dir = Path('data/single-document/BBC News Summary/text/politics')
summaries_dir = Path('data/single-document/BBC News Summary/summaries/politics')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

for file_path in text_dir.files('*.txt'):
    with file_path.open(mode='rt', encoding='utf-8') as fp:
        texts.append(fp.readlines())

lxr = LexRank(texts, stopwords=STOPWORDS['en'])

scores = {}

for i in range(1, 10):
    text_file_path = text_dir + '/00' + str(i) + '.txt'
    with text_file_path.open(mode='rt', encoding='utf-8') as fp:
        # f = open("data/single-document/BBC News Summary/text/politics/001.txt", "r")
        sentences = tokenizer.tokenize(fp.read())

        # get summary with continuous LexRank
        summarySentences = lxr.get_summary(sentences, summary_size=8, threshold=None, include_keyphrase_similarity = False, redunduncy_penalty = True)

    summary_file_path = summaries_dir + '/00' + str(i) + '.txt'
    with summary_file_path.open(mode='rt', encoding='utf-8') as fp:
        modelSummarySentences = tokenizer.tokenize(fp.read())

    producedSummary = ' '.join(summarySentences)
    modelSummary = ' '.join(modelSummarySentences)

    score = rouge_1(producedSummary, modelSummary)
    scores[i] = score

    # print(producedSummary)
    # print('=' * 20)
    # print(modelSummary)
    # print('=' * 20)
    # print(score)

print(scores)
