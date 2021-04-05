import os
from os.path import join, getsize
import re
from utils import STOPWORDS, LexRank
from path import Path
from myRouge import rouge_1
import rouge
import nltk.data
from keyPhraseExtractor import KeyPhraseExtractor

texts = []
text_dir = Path('data/multi-document/DUC 2003/text')
summaries_dir = Path('data/multi-document/DUC 2003/summaries')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

documentClusterSize = 0
documentClusterPaths= []

for root, dirs, files in os.walk(text_dir):
    if dirs != []:
        for dir in dirs:
            curr_dir = text_dir + '/' + dir
            new_file = join(curr_dir, 'merge.txt')
            fp = open(new_file,"r")
            texts.append(fp.readlines())


lxr = LexRank(texts, stopwords=STOPWORDS['en'])

scores = {}
avgScore = 0.0
dirCount = 0
for root, dirs, files in os.walk(text_dir):
    if dirs != []:
        for dir in dirs:
            if dirCount >= 3:
                break
            dirCount += 1
            curr_dir = text_dir + '/' + dir
            text_file_path = join(curr_dir, 'merge.txt')
            f = open(text_file_path,"r")

            sentences = tokenizer.tokenize(f.read())

            summarySentences = lxr.get_summary(sentences, summary_size=6, threshold=None, include_keyphrase_similarity = False, redunduncy_penalty = True)

            producedSummary = ' '.join(summarySentences)

            result = re.search('\D*(\d*)\D*', dir)
            dirNum = result.group(1)
            score = 0
            count = 0
            for summary_file_path in summaries_dir.files('D' + dirNum + '.M.100.*'):
                with summary_file_path.open(mode='rt', encoding='utf-8') as fp:
                    count += 1
                    modelSummarySentences = tokenizer.tokenize(fp.read())
                    modelSummary = ' '.join(modelSummarySentences)
                    score += rouge_1(producedSummary, modelSummary)
            score /= count
            scores[dir] = score
            avgScore += score

            # print(producedSummary)
            # print('=' * 20)
            # print(modelSummary)
            # print('=' * 20)
            # print(score)

avgScore /= len(scores)
print(avgScore)
# print(scores)
