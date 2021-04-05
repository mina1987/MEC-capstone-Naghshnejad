from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


def stemAndRemoveStopWords(originalText):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(originalText)
    ps = PorterStemmer()
    newWords = []
    for word in word_tokens:
        stemmedWord = ps.stem(word)
        if stemmedWord not in stop_words:
            newWords.append(stemmedWord)
    return newWords

def rouge_1(producedSummary, modelSummary):
    cleanedModelSummaryWords = stemAndRemoveStopWords(modelSummary)
    cleanedProducedSummaryWords = stemAndRemoveStopWords(producedSummary)

    sizeOfModelSummary = len(cleanedModelSummaryWords)
    count = 0.0

    for word in cleanedModelSummaryWords:
        if word in cleanedProducedSummaryWords:
            count += 1
    return count/sizeOfModelSummary
