from summarizer import LexRank
import numpy as np
class USERank(LexRank):
    def __init__(
        self,
        documents,
        model,   
        stopwords=None,
        keep_numbers=False,
        keep_emails=False,
        keep_urls=False,
        include_new_words=True
    ):
        self.documents=documents
        if stopwords is None:
            self.stopwords = set()
        else:
            self.stopwords = stopwords

        self.keep_numbers = keep_numbers
        self.keep_emails = keep_emails
        self.keep_urls = keep_urls
        self.include_new_words = include_new_words
        
    def get_summary(
        self,
        model,
        sentences,
        summary_size=1,
        threshold=.03,
        fast_power_method=True,
        redunduncy_penalty = False,
        include_keyphrase_similarity = False,
        d = 0,
    ):
        if not isinstance(summary_size, int) or summary_size < 1:
            raise ValueError('\'summary_size\' should be a positive integer')

        use_scores = self.rank_sentences(
            model,
            sentences,
            threshold=threshold,
            fast_power_method=fast_power_method,
        )
        

        sorted_ix = np.argsort(use_scores)[::-1]
        summary = [sentences[i] for i in sorted_ix[:summary_size]]

        if redunduncy_penalty:
            S2 = [(sentences[i], use_scores[i]) for i in sorted_ix]
            summary = self.addRedunduncyPenalty(S2, summary_size)

        return summary

    def rank_sentences(
        self,
        model,
        sentences,
        threshold=.03,
        fast_power_method=True,
    ):
        if not (
            threshold is None or
            isinstance(threshold, float) and 0 <= threshold < 1
        ):
            raise ValueError(
                '\'threshold\' should be a floating-point number '
                'from the interval [0, 1) or None',
            )

   

        similarity_matrix = self._calculate_similarity_matrix(model)

        if threshold is None:
            markov_matrix = self._markov_matrix(similarity_matrix)

        else:
            markov_matrix = self._markov_matrix_discrete(
                similarity_matrix,
                threshold=threshold,
            )

        scores = stationary_distribution(
            markov_matrix,
            increase_power=fast_power_method,
            normalized=False,
        )

        return scores

    def cosine(self, u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))    
        
    def _calculate_similarity_matrix(self, model):
        length = len(self.documents)

        similarity_matrix = np.zeros([length] * 2)

        for i in range(length):
            for j in range(i, length):
                similarity =  self.cosine(model([self.documents[i]])[0], model([self.documents[j]])[0])
                if similarity:
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        return similarity_matrix
