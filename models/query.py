from typing import List

from torch import tensor

from utils.text_transformer import TextTransformer


class Query:
    def __init__(self, content, stopwords_list, text_transformer: TextTransformer):
        self.content = content
        self.__stopwords = stopwords_list
        self.__text_transformer = text_transformer
        self.tokens: List[str] = []
        self.__length = len(self.tokens)
        self.term_frequencies = {}
        self.embedding: tensor = tensor([])
        self.__process_query()

    def __tokenize(self):
        self.tokens = self.__text_transformer.tokenize(self.content)

    def __remove_stopwords(self):
        self.tokens = [token for token in self.tokens if token not in self.__stopwords]
        self.__length = len(self.tokens)

    def __lemmatize(self):
        self.tokens = self.__text_transformer.lemmatize(" ".join(self.tokens))

    def __get_term_frequencies(self):
        for token in self.tokens:
            if token in self.term_frequencies:
                self.term_frequencies[token] += 1
            else:
                self.term_frequencies[token] = 1

    def __process_query(self):
        self.__tokenize()
        self.__remove_stopwords()
        self.__lemmatize()
        self.__get_term_frequencies()
        if self.__text_transformer.context_retrieval:
            self.__get_embedding()

    def get_tf(self, target_term):
        try:
            tf = self.term_frequencies[target_term]
        except KeyError:
            return 0
        return tf

    def get_vocabulary(self):
        return list(self.term_frequencies.keys())

    def __get_embedding(self):
        self.embedding = self.__text_transformer.encode(self.content)
