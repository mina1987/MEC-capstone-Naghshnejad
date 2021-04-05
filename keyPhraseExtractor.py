import pke
import math
from nltk.corpus import wordnet as wn
from utils.utils.text import tokenize
from collections import Counter



class KeyPhraseExtractor:
    def __init__(self, idf_score, language = 'en', numKeyPhrases = 10, stopwords = None):
        self.language = language
        self.numKeyPhrases = numKeyPhrases
        self.idf_score = idf_score
        self.stopwords = stopwords


    def getKeyPhrases(self, text_file_path, withScores = False):
        # initialize keyphrase extraction model, here TopicRank
        extractor = pke.unsupervised.TopicRank()

        # load the content of the document, here document is expected to be in raw
        # format (i.e. a simple text file) and preprocessing is carried out using spacy
        extractor.load_document(input=text_file_path, language=self.language)

        # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
        # and adjectives (i.e. `(Noun|Adj)*`)
        extractor.candidate_selection()

        # candidate weighting, in the case of TopicRank: using a random walk algorithm
        extractor.candidate_weighting()

        # N-best selection, keyphrases contains the 10 highest scored candidates as
        # (keyphrase, score) tuples
        keyphrases = extractor.get_n_best(n=self.numKeyPhrases)

        if withScores:
            return keyphrases
        return [keyphrase for (keyphrase, score) in keyphrases]

    def sentenceKeyphraseSimilarity(self, keyphrase, sentence):
        tf_1 = Counter(self.tokenizeSentence(keyphrase))
        tf_2 = Counter(self.tokenizeSentence(sentence))

        similarity = self.cosineSimilarity([tf_1, tf_2], 0, 1)

        return similarity

    def tokenizeSentence(self, sentence):
        tokens = tokenize(
            sentence,
            self.stopwords,
            keep_numbers = False,
            keep_emails = False,
            keep_urls = False,
        )

        return tokens

    def cosineSimilarity(self, tf_scores, i, j):
        if i == j:
            return 1

        tf_i, tf_j = tf_scores[i], tf_scores[j]
        words_i, words_j = set(tf_i.keys()), set(tf_j.keys())

        nominator = 0

        for word in words_i & words_j:
            idf = self.idf_score[word]
            nominator += tf_i[word] * tf_j[word] * idf ** 2

        if math.isclose(nominator, 0):
            return 0

        denominator_i, denominator_j = 0, 0

        for word in words_i:
            tfidf = tf_i[word] * self.idf_score[word]
            denominator_i += tfidf ** 2

        for word in words_j:
            tfidf = tf_j[word] * self.idf_score[word]
            denominator_j += tfidf ** 2

        similarity = nominator / math.sqrt(denominator_i * denominator_j)

        return similarity

    def getKeyPhraseSynonyms(self, keyphrases):
        synonyms = {}
        for keyphrase in keyphrases:
            listOfSynonyms = [keyphrase]
            modifiedKeyphrase = keyphrase.replace(' ', '_')
            for synset in wn.synsets(modifiedKeyphrase):
                for lemma in synset.lemmas():
                    listOfSynonyms.append(str(lemma.name()))
            listOfSynonyms = list(dict.fromkeys(listOfSynonyms))
            synonyms[keyphrase] = listOfSynonyms
        return synonyms

    def getKeyPhraseSentencesSimilarity(self, text_file_path, sentences, approach = 1, withSynonyms = False):
        if approach == 1:
            keyphrase_scores = []
            keyphrases = self.getKeyPhrases(text_file_path)
            if withSynonyms:
                keyphrasesAndSynonyms = self.getKeyPhraseSynonyms(keyphrases)
                for sentence in sentences:
                    count = 0.0
                    for keyphrase in keyphrases:
                        for synonym in keyphrasesAndSynonyms[keyphrase]:
                            if synonym in sentence:
                                count += 1
                    keyphrase_scores.append(count)
                return keyphrase_scores
            for sentence in sentences:
                count = 0.0
                for keyphrase in keyphrases:
                    if keyphrase in sentence:
                        count += 1
                keyphrase_scores.append(count)
            return keyphrase_scores
        if approach == 2:
            keyphrase_scores = []
            keyphrases = self.getKeyPhrases(text_file_path)
            if withSynonyms:
                keyphrasesAndSynonyms = self.getKeyPhraseSynonyms(keyphrases)
                for sentence in sentences:
                    count = 0.0
                    for keyphrase in keyphrases:
                        for synonym in keyphrasesAndSynonyms[keyphrase]:
                            count += self.sentenceKeyphraseSimilarity(synonym, sentence)
                    keyphrase_scores.append(count)
                return keyphrase_scores
            for sentence in sentences:
                count = 0.0
                for keyphrase in keyphrases:
                    count += self.sentenceKeyphraseSimilarity(keyphrase, sentence)
                keyphrase_scores.append(count)
            return keyphrase_scores
        if approach == 3:
            keyphrase_scores = []
            keyphrases = self.getKeyPhrases(text_file_path, withScores = True)
            if withSynonyms:
                keyphrasesWithoutScores = [keyphrase for (keyphrase, score) in keyphrases]
                keyphrasesAndSynonyms = self.getKeyPhraseSynonyms(keyphrasesWithoutScores)
                for sentence in sentences:
                    count = 0.0
                    for (keyphrase, score) in keyphrases:
                        for synonym in keyphrasesAndSynonyms[keyphrase]:
                            if synonym in sentence:
                                count += score
                    keyphrase_scores.append(count)
                return keyphrase_scores
            for sentence in sentences:
                count = 0.0
                for (keyphrase, score) in keyphrases:
                    if keyphrase in sentence:
                        count += 100 * score
                keyphrase_scores.append(count)
            return keyphrase_scores
