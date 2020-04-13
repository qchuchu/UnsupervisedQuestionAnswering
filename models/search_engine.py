from math import sqrt
from typing import List, Tuple
from pickle import load

import click
from torch.nn import CosineSimilarity

from models.collection import Collection
from models.query import Query
from utils.helpers import merge_or_postings_list
from utils.text_transformer import TextTransformer


class SearchEngine:
    def __init__(
        self,
        collection_name: str,
        stopwords_list,
        text_transformer: TextTransformer,
        weighting_model: str = "tw-idf",
    ):
        self.collection = Collection(
            collection_name, stopwords_list, weighting_model, text_transformer
        )
        self.weighting_model = weighting_model
        self.stopwords = stopwords_list
        self.__text_transformer = text_transformer
        self.__cos = CosineSimilarity(dim=0, eps=1e-6)

    def search(self, string_query: str, article_window: int = 10):
        query = Query(string_query, self.stopwords, self.__text_transformer)
        posting_list = self.__get_posting_list(query)
        doc_scores = self.__get_doc_scores(posting_list, query)
        if self.__text_transformer.context_retrieval:
            sorted_docs = [
                (k, v)
                for k, v in sorted(
                    doc_scores.items(), key=lambda item: item[1], reverse=True
                )
            ]
            context_scores = self.__get_context_scores(
                sorted_docs[:article_window], query
            )
            return context_scores
        return doc_scores

    def __get_posting_list(self, query: Query):
        final_posting_list = []
        vocabulary = query.get_vocabulary()
        for token in vocabulary:
            if not final_posting_list:
                final_posting_list = self.collection.get_posting_list(token)
            else:
                posting_list = self.collection.get_posting_list(token)
                final_posting_list = merge_or_postings_list(
                    final_posting_list, posting_list
                )
        return final_posting_list

    def __get_doc_scores(self, posting_list, query: Query):
        click.secho("[Search Engine] Computing search scores ...", fg="bright_blue")
        query_tf_idf = {}
        norm_query_vector = 0
        query_vocabulary = query.get_vocabulary()
        for token in query_vocabulary:
            tf_idf = query.get_tf(token) * self.collection.get_idf(token)
            query_tf_idf[token] = tf_idf
            norm_query_vector += tf_idf ** 2
        norm_query_vector = sqrt(norm_query_vector)
        doc_scores = {}
        for doc_id in posting_list:
            score = 0
            for token in query_vocabulary:
                if self.weighting_model == "tw-idf":
                    weight = self.collection.get_tw_idf(
                        target_term=token, target_doc_id=doc_id, b=0.003
                    )
                elif self.weighting_model == "tf-idf":
                    weight = self.collection.get_piv_plus(
                        target_term=token, target_doc_id=doc_id, b=0.2
                    )
                else:
                    weight = self.collection.get_bm25_plus(
                        target_term=token, target_doc_id=doc_id, b=0.75, k1=1.2
                    )
                score += query_tf_idf[token] * weight
            score /= self.collection.documents_norms[doc_id] * norm_query_vector
            doc_scores[doc_id] = score
        return doc_scores

    def __get_context_scores(
        self, selected_docs: List[Tuple[int, float]], query: Query
    ):
        context_scores = {}
        query_embedding = query.embedding
        for (doc_id, doc_score) in selected_docs:
            document = self.collection.documents[doc_id]
            url_without_extension = document.url.split(".")[0]
            for id_context in range(document.nb_contexts):
                context_filename = "contexts/{}_{}_{}_{}.p".format(
                    self.__text_transformer.embedding_model_label,
                    self.__text_transformer.pooling_modes,
                    url_without_extension,
                    id_context,
                )
                with open(context_filename, "rb") as pickle_context:
                    context = load(pickle_context)
                score = self.__cos(query_embedding, context.embedding).item()
                context_scores[
                    (doc_id, "{}_{}".format(url_without_extension, id_context))
                ] = (score * doc_score)
        return context_scores


if __name__ == "__main__":
    text_transformer = TextTransformer()
    search_engine = SearchEngine(
        collection_name="fquad", stopwords_list=[], text_transformer=text_transformer
    )
    scores = search_engine.search(
        "Pourquoi Cérès n'était pas directement assimilable à une comète ?"
    )
    print(scores)
