from os import path
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from typing import List
from pickle import dump

from utils.text_transformer import TextTransformer
from utils.helpers import create_graph_of_words
from models.context import Context

STOPWORDS = stopwords.words("french")


class Document:
    """
    This is class is for representing a document from the CS276 dataset
    """

    def __init__(self, id_doc: int, url: str, folder: int):
        self.__id = id_doc
        self.__folder = folder
        self.__title: str = ""
        self.__url: str = url
        self.tokens: List[str] = []
        self.content = ""
        self.__key_words: List[str] = []
        self.__length: int = 0
        self.__nb_contexts: int = 0

    @property
    def length(self):
        return self.__length

    @property
    def id(self):
        return self.__id

    @property
    def folder(self):
        return self.__folder

    @property
    def url(self):
        return self.__url

    @property
    def title(self):
        return self.__title

    @property
    def key_words(self):
        return self.__key_words

    @property
    def nb_contexts(self):
        return self.__nb_contexts

    def load_data(self, path_to_documents: str, text_transformer: TextTransformer):
        path_to_file = path.join(
            path_to_documents, "{}/{}".format(self.__folder, self.__url)
        )
        with open(path_to_file, "r") as file:
            if text_transformer.context_retrieval:
                title = next(file)
                self.__title = title.rstrip("\n")
                for id_context, line in enumerate(file):
                    content = line.rstrip("\n")
                    self.__store_context(id_context, content, text_transformer)
                    self.tokens.extend(text_transformer.tokenize(content))
            else:
                self.content = next(file).rstrip("\n")
                self.tokens.extend(text_transformer.tokenize(self.content))
        self.__remove_not_alpha()

    def __store_context(
        self, id_context: int, content: str, text_transformer: TextTransformer
    ):
        context = Context(content)
        context.encode(text_transformer)
        url_without_extension = self.url.split(".")[0]
        context_pickle_name = "{}_{}_{}_{}.p".format(
            text_transformer.embedding_model_label,
            text_transformer.pooling_modes,
            url_without_extension,
            id_context,
        )
        with open("contexts/{}".format(context_pickle_name), "wb") as target_file:
            dump(context, target_file)
        self.__nb_contexts += 1

    def __remove_not_alpha(self):
        filtered_tokens = []
        for token in self.tokens:
            if token.isalpha():
                filtered_tokens.append(token)
        self.tokens = filtered_tokens
        self.__length = len(self.tokens)

    def __store_key_words(self):
        tokens_without_stopwords = [
            token for token in self.tokens if token not in STOPWORDS
        ]
        counter = Counter(tokens_without_stopwords)
        self.__key_words = [x[0] for x in counter.most_common(5)]

    def __remove_stopwords(self, stopwords_list):
        self.tokens = [token for token in self.tokens if token not in stopwords_list]
        self.__length = len(self.tokens)

    def __lemmatize(self, text_transformer: TextTransformer):
        self.tokens = text_transformer.lemmatize(" ".join(self.tokens))

    def get_term_weights(self):
        graph_of_words = create_graph_of_words(window=4, tokens=self.tokens)
        term_weights = {}
        for term, indegree_edges in graph_of_words.items():
            term_weights[term] = len(indegree_edges)
        return term_weights

    def get_term_frequencies(self):
        term_frequencies = {}
        for token in self.tokens:
            if token in term_frequencies:
                term_frequencies[token] += 1
            else:
                term_frequencies[token] = 1
        return term_frequencies

    def process_document(self, stopwords_list, text_transformer: TextTransformer):
        self.__remove_stopwords(stopwords_list)
        self.__store_key_words()
        self.__lemmatize(text_transformer)

    def get_vocabulary(self):
        return list(set(self.tokens))


if __name__ == "__main__":
    text_transformer = TextTransformer()
    document = Document(url="1_cérès.txt", folder=0, id_doc=0)
    document.load_data(
        path_to_documents="data/fquad", text_transformer=text_transformer
    )
    document.process_document([], text_transformer)
    print(document.nb_contexts)
    print(document.title)
    print(document.tokens)
