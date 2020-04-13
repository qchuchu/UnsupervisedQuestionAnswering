from os import path, listdir, getcwd, walk
from pickle import load, dump
from typing import List, Dict
from math import log, sqrt

from tqdm import tqdm
import click

from models.document import Document
from utils.text_transformer import TextTransformer

WEIGHTING_MODELS = ["tf-idf", "tw-idf", "okapi-bm25"]
WEIGHTING_MODEL_INDEX = {"tf-idf": "tf", "tw-idf": "tw", "okapi-bm25": "tf"}


class WeightingModelError(Exception):
    pass


class Collection:
    """
    This class will represent a collection of documents
    """

    def __init__(
        self,
        name: str,
        stopwords_list: List[str],
        weighting_model: str,
        text_transformer: TextTransformer,
    ):
        weighting_model = weighting_model.lower()
        if weighting_model not in WEIGHTING_MODELS:
            raise WeightingModelError(
                "{} is not an available weighting model".format(weighting_model)
            )
        self.__name = name
        self.documents: List[Document] = []
        self.inverted_index: Dict[str, Dict[int, int]] = {}
        self.documents_norms: Dict[int, float] = {}
        self.stopwords = stopwords_list
        self.path_to_data = path.join(getcwd(), "data/{}".format(name))
        self.nb_docs = sum([len(files) for r, d, files in walk(self.path_to_data)])
        self.nb_folders = sum([len(d) for r, d, files in walk(self.path_to_data)])
        self.average_document_length = 0
        self.text_transformer = text_transformer
        click.secho(
            "[Collection] The chosen index is based on : {}".format(
                weighting_model.upper(), fg="bright_blue"
            )
        )
        click.secho("[Collection] Loading Documents...", fg="bright_blue")
        self.__load_documents()
        click.secho("[Collection] All Document Loaded !", fg="bright_blue")
        click.secho(
            "[Collection] Loading {} Inverted Index...".format(weighting_model.upper()),
            fg="bright_blue",
        )
        self.__load_inverted_index(weighting_model)
        click.secho("[Collection] {} Inverted Index Loaded !", fg="bright_blue")
        click.secho("[Collection] Load Documents Norms...", fg="bright_blue")
        self.__load_documents_norms(weighting_model)
        click.secho("[Collection] Documents Norms Loaded !", fg="bright_blue")

    @property
    def name(self):
        return self.__name

    def __load_documents(self):
        try:
            if self.text_transformer.context_retrieval:
                self.documents = self.__load_pickle_file(
                    "{}_{}_preprocessed_documents".format(
                        self.text_transformer.embedding_model_label,
                        self.text_transformer.pooling_modes,
                    )
                )
            else:
                self.documents = self.__load_pickle_file("preprocessed_documents")
        except FileNotFoundError:
            nb_document_loaded = 0
            for directory_index in range(self.nb_folders):
                click.secho(
                    "[Collection] Processing folder #{}...".format(directory_index),
                    fg="bright_blue",
                )
                path_directory = path.join(self.path_to_data, str(directory_index))
                for filename in tqdm(listdir(path_directory)):
                    document = Document(
                        url=filename, folder=directory_index, id_doc=nb_document_loaded
                    )
                    document.load_data(self.path_to_data, self.text_transformer)
                    document.process_document(
                        stopwords_list=self.stopwords,
                        text_transformer=self.text_transformer,
                    )
                    self.documents.append(document)
                    nb_document_loaded += 1
            if self.text_transformer.context_retrieval:
                self.__store_pickle_file(
                    "{}_{}_preprocessed_documents".format(
                        self.text_transformer.embedding_model_label,
                        self.text_transformer.pooling_modes,
                    ),
                    self.documents,
                )
            else:
                self.__store_pickle_file("preprocessed_documents", self.documents)
        assert len(self.documents) == self.nb_docs
        for document in self.documents:
            self.average_document_length += document.length
        self.average_document_length /= self.nb_docs

    def __load_inverted_index(self, weighting_model: str):
        inverted_index_model = WEIGHTING_MODEL_INDEX[weighting_model]
        pickle_filename = "inverted_index_{}".format(inverted_index_model)
        try:
            self.inverted_index = self.__load_pickle_file(pickle_filename)
        except FileNotFoundError:
            click.secho("[Collection] Creating inverted index ...", fg="bright_blue")
            for document in tqdm(self.documents):
                if inverted_index_model == "tw":
                    term_weights = document.get_term_weights()
                else:
                    term_weights = document.get_term_frequencies()
                for term, weight in term_weights.items():
                    if term in self.inverted_index:
                        self.inverted_index[term][document.id] = weight
                    else:
                        self.inverted_index[term] = {document.id: weight}
            self.__store_pickle_file(pickle_filename, self.inverted_index)

    def __load_documents_norms(self, weighting_model: str):
        pickle_filename = "document_norms_{}".format(weighting_model)
        try:
            self.documents_norms = self.__load_pickle_file(pickle_filename)
        except FileNotFoundError:
            click.echo("[Collection] Computing documents norm ...")
            nb_norms_calculated = 0
            for document in tqdm(self.documents):
                doc_vocabulary = document.get_vocabulary()
                norm = 0
                for token in doc_vocabulary:
                    if weighting_model == "tw-idf":
                        norm += self.get_tw_idf(token, document.id, 0.003) ** 2
                    elif weighting_model == "tf-idf":
                        norm += self.get_piv_plus(token, document.id, 0.2) ** 2
                    else:
                        norm += self.get_bm25_plus(token, document.id, 0.75, 1.2) ** 2
                norm = sqrt(norm)
                self.documents_norms[document.id] = norm
                nb_norms_calculated += 1
            self.__store_pickle_file(pickle_filename, self.documents_norms)

    def __load_pickle_file(self, filename):
        pickle_filename = "indexes/{}_{}.p".format(self.name, filename)
        return load(open(pickle_filename, "rb"))

    def __store_pickle_file(self, filename, collection_object):
        target_file = open("indexes/{}_{}.p".format(self.name, filename), "wb")
        dump(collection_object, target_file)

    def get_vocabulary(self):
        return list(self.inverted_index.keys())

    def __get_term_weight(self, target_term, target_doc_id):
        try:
            term_weight = self.inverted_index[target_term][target_doc_id]
            return term_weight
        except KeyError:
            return 0

    def __get_term_frequency(self, target_term, target_doc_id):
        try:
            term_frequency = self.inverted_index[target_term][target_doc_id]
            return term_frequency
        except KeyError:
            return 0

    def __get_pivoted_normalizer(self, target_doc_id, b):
        pivoted_normalizer = (
            1
            - b
            + (b * self.documents[target_doc_id].length / self.average_document_length)
        )
        return pivoted_normalizer

    def __get_pivoted_term_weight(self, target_term, target_doc_id, b):
        term_weight = self.__get_term_weight(target_term, target_doc_id)
        if term_weight == 0:
            return 0
        return term_weight / self.__get_pivoted_normalizer(target_doc_id, b)

    def __get_pivoted_and_concave_ln_tf(self, target_term, target_doc_id, b):
        term_frequency = self.__get_term_frequency(target_term, target_doc_id)
        if term_frequency == 0:
            return 0
        concave_tf = 1 + log(1 + log(term_frequency))
        return concave_tf / self.__get_pivoted_normalizer(target_doc_id, b)

    def __get_pivoted_and_concave_k_tf(self, target_term, target_doc_id, b, k1):
        term_frequency = self.__get_term_frequency(target_term, target_doc_id)
        if term_frequency == 0:
            return 0
        concave_tf = (k1 + 1) * term_frequency
        K = k1 * self.__get_pivoted_normalizer(target_doc_id, b)
        return concave_tf / (K + term_frequency)

    def get_idf(self, target_term):
        try:
            df = len(self.inverted_index[target_term].keys())
        except KeyError:
            return 0
        return log((self.nb_docs + 1) / df)

    def get_tw_idf(self, target_term, target_doc_id, b):
        return self.__get_pivoted_term_weight(
            target_term, target_doc_id, b
        ) * self.get_idf(target_term)

    def get_tf_idf(self, target_term, target_doc_id, b):
        return self.__get_pivoted_and_concave_ln_tf(
            target_term, target_doc_id, b
        ) * self.get_idf(target_term)

    def get_piv_plus(self, target_term, target_doc_id, b):
        normalized_tf = self.__get_pivoted_and_concave_ln_tf(
            target_term, target_doc_id, b
        )
        if normalized_tf == 0:
            return 0
        else:
            return (normalized_tf + 1) * self.get_idf(target_term)

    def get_bm25(self, target_term, target_doc_id, b, k1):
        return self.__get_pivoted_and_concave_k_tf(
            target_term, target_doc_id, b, k1
        ) * self.get_idf(target_term)

    def get_bm25_plus(self, target_term, target_doc_id, b, k1):
        normalized_tf = self.__get_pivoted_and_concave_k_tf(
            target_term, target_doc_id, b, k1
        )
        if normalized_tf == 0:
            return 0
        else:
            return (normalized_tf + 1) * self.get_idf(target_term)

    def get_posting_list(self, target_term):
        try:
            doc_list = list(self.inverted_index[target_term].keys())
        except KeyError:
            return []
        return doc_list


if __name__ == "__main__":
    text_transformer = TextTransformer()
    collection = Collection(
        name="fquad",
        stopwords_list=[],
        weighting_model="tw-idf",
        text_transformer=text_transformer,
    )
