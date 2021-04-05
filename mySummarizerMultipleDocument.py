import os
from os.path import join, getsize
import re
from utils import STOPWORDS, LexRank
from path import Path
from myRouge import rouge_1
import rouge
import nltk.data
from keyPhraseExtractor import KeyPhraseExtractor
import matplotlib.pyplot as plt


texts = []

# for DUC 2003
# text_dir = Path('data/multi-document/DUC 2003/text')
# summaries_dir = Path('data/multi-document/DUC 2003/summaries')

# for DUC 2004 10-word model summaries
text_dir = Path('data/multi-document/DUC 2004/text')
summaries_dir = Path('data/multi-document/DUC 2004/summaries')

# for DUC 2004 100-word model summaries
# text_dir = Path('data/multi-document/DUC 2004/text')
# summaries_dir = Path('data/multi-document/DUC 2004/summaries2')


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
kphExtractor = KeyPhraseExtractor(lxr.idf_score, stopwords=STOPWORDS['en'])

# d = [x/ 10.0 for x in range(11)]
# options = [(1, True, True, "Approach #1 with synonyms (and with redunduncy penalty)"), (1, False, True, "Approach #1 without synonyms (and with redunduncy penalty)"), \
# (2, True, True, "Approach #2 with synonyms (and with redunduncy penalty)"), (2, False, True, "Approach #2 without synonyms (and with redunduncy penalty)"), \
# (3, True, True, "Approach #3 with synonyms (and with redunduncy penalty)"), (3, False, True, "Approach #3 without synonyms (and with redunduncy penalty)")]

d = [0, 0.4]
options = [(1, True, True, "Approach #1 with synonyms and with redunduncy penalty")]
for i, option in enumerate(options):
    maxD = None
    maxPercentage = None
    avgRScoreForD = []
    for dVal in d:
        scores = {}
        avgScore = 0.0
        for root, dirs, files in os.walk(text_dir):
            if dirs != []:
                for dir in dirs:
                    curr_dir = text_dir + '/' + dir
                    text_file_path = join(curr_dir, 'merge.txt')
                    f = open(text_file_path,"r")

                    sentences = tokenizer.tokenize(f.read())

                    keyphrase_scores = kphExtractor.getKeyPhraseSentencesSimilarity(text_file_path, sentences, approach = option[0],  withSynonyms = option[1])

                    summarySentences = lxr.get_summary(sentences, summary_size=1, threshold=None, include_keyphrase_similarity = True, keyphrase_similarity_scores = keyphrase_scores, d = dVal, redunduncy_penalty = option[2])

                    producedSummary = ' '.join(summarySentences)

                    result = re.search('\D*(\d*)\D*', dir)
                    dirNum = result.group(1)
                    score = 0
                    count = 0

                    # for the 100 word summaries of DUC 2003
                    # for summary_file_path in summaries_dir.files('D' + dirNum + '.M.100.*'):

                    # for the 100 word summaries of DUC 2004
                    # summaries_curr_dir = summaries_dir + '/' + dir
                    # for summary_file_path in summaries_curr_dir.files('D' + dirNum + '.M.100.*'):

                    # for the 10 word summaries of DUC 2004
                    for summary_file_path in summaries_dir.files('D' + dirNum + '.P.10.*'):
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
        if maxPercentage is None or avgScore > maxPercentage:
            maxPercentage = avgScore
            maxD = dVal
        avgRScoreForD.append(avgScore)
        # print(avgScore)

    print("plotting!")
    print("Max D: ", maxD)
    print("With percentage: ", maxPercentage)
    plt.figure(i + 1)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.grid()
    plt.plot(d, avgRScoreForD, 'ro')
    plt.xlabel('Value of d')
    plt.ylabel('Average ROUGE-1 Score')
    plt.title(option[3])
    print(avgRScoreForD)


plt.show()
