import click

from models.search_engine import SearchEngine
from utils.text_transformer import TextTransformer


@click.command()
@click.option(
    "-n", "--collection_name", default="fquad", help="Collection Name", type=str
)
@click.option(
    "-w",
    "--weighting_model",
    default="tw-idf",
    type=click.Choice(["tw-idf", "tf-idf", "okapi-bm25"], case_sensitive=False),
    help="Weighting Type.",
)
@click.option(
    "-m",
    "--embedding_model",
    default="camembert-base",
    help="Context Embedding Model",
    type=str,
)
@click.option("-l", "--lemmatizer", default="spacy-fr", help="Lemmatizer used")
@click.option(
    "-M",
    "--mean_tokens",
    default=True,
    help="Taking into account the mean of the embedding of each token",
    type=bool,
)
@click.option(
    "-C",
    "--cls_token",
    default=True,
    help="Taking into account the CLS token of the phrase",
    type=bool,
)
@click.option(
    "-X",
    "--max_tokens",
    default=False,
    help="Taking into account the max of the embedding of each token",
    type=bool,
)
@click.option(
    "-d",
    "--document_tokenizer",
    default="only-words",
    help="How to tokenize the text for the inverted index",
    type=str,
)
@click.option(
    "-r",
    "--context_retrieval",
    default=True,
    help="If you want to make a deeper search on the contexts inside a document or not",
    type=bool,
)
def get_accuracy_fquad(
    collection_name,
    weighting_model,
    embedding_model,
    lemmatizer,
    mean_tokens,
    cls_token,
    max_tokens,
    document_tokenizer,
    context_retrieval,
):
    # We get all queries
    with open("dev_resources/queries/fquad.in", "r") as query_file:
        queries = []
        for query in query_file:
            queries.append(query.rstrip("\n"))
    # We get all answers
    with open("dev_resources/output/fquad.out", "r") as answer_file:
        answers = []
        for answer in answer_file:
            answers.append(answer.rstrip("\n"))
    text_transformer = TextTransformer(
        lemmatizer_label=lemmatizer,
        embedding_model_label=embedding_model,
        document_tokenizer=document_tokenizer,
        mean_tokens=mean_tokens,
        cls_token=cls_token,
        max_tokens=max_tokens,
        context_retrieval=context_retrieval,
    )
    search_engine = SearchEngine(
        collection_name=collection_name,
        stopwords_list=[],
        text_transformer=text_transformer,
        weighting_model=weighting_model,
    )
    found_context_top_k = [0 for _ in range(25)]
    found_article_top_k = [0 for _ in range(25)]
    for id_query, query in enumerate(queries):
        scores_query = search_engine.search(query)
        total_results = min(len(scores_query.keys()), 25)
        sorted_results = [
            k
            for k, v in sorted(
                scores_query.items(), key=lambda item: item[1], reverse=True
            )
        ][:total_results]
        is_answer_found = False
        is_article_found = False
        id_result = 0
        target_context = answers[id_query]
        target_article = "_".join(target_context.split("_")[:-1])
        while not is_answer_found and id_result < total_results:
            if text_transformer.context_retrieval:
                doc_id_query, context_name = sorted_results[id_result]
            else:
                doc_id_query = sorted_results[id_result]
                document = search_engine.collection.documents[doc_id_query]
                context_name = document.url.split(".")[0]
            if context_name == target_context:
                is_answer_found = True
                found_context_top_k[id_result] += 1
            if not is_article_found:
                article = "_".join(context_name.split("_")[:-1])
                if article == target_article:
                    is_article_found = True
                    found_article_top_k[id_result] += 1
            id_result += 1
    nb_queries = len(queries)
    total_article_score = 0
    total_context_score = 0
    if not text_transformer.context_retrieval:
        with open(
            "results/{}_{}_context_scores.txt".format(collection_name, weighting_model),
            "w",
        ) as result_file:
            for k, score in enumerate(found_context_top_k):
                total_context_score += score
                result = "{} {}".format(
                    k + 1, "{0:.2f}".format(total_context_score / nb_queries * 100)
                )
                print(result)
                result_file.write(f"{result}\n")
        with open(
            "results/{}_{}_article_scores.txt".format(collection_name, weighting_model),
            "w",
        ) as result_file:
            for k, score in enumerate(found_article_top_k):
                total_article_score += score
                result = "{} {}".format(
                    k + 1, "{0:.2f}".format(total_article_score / nb_queries * 100)
                )
                print(result)
                result_file.write(f"{result}\n")
    else:
        with open(
            "results/{}_{}_{}_{}_context_scores.txt".format(
                embedding_model,
                text_transformer.pooling_modes,
                collection_name,
                weighting_model,
            ),
            "w",
        ) as result_file:
            for k, score in enumerate(found_context_top_k):
                total_context_score += score
                result = "{} {}".format(
                    k + 1, "{0:.2f}".format(total_context_score / nb_queries * 100)
                )
                print(result)
                result_file.write(f"{result}\n")
        with open(
            "results/{}_{}_{}_{}_article_scores.txt".format(
                embedding_model,
                text_transformer.pooling_modes,
                collection_name,
                weighting_model,
            ),
            "w",
        ) as result_file:
            for k, score in enumerate(found_article_top_k):
                total_article_score += score
                result = "{} {}".format(
                    k + 1, "{0:.2f}".format(total_article_score / nb_queries * 100)
                )
                print(result)
                result_file.write(f"{result}\n")


if __name__ == "__main__":
    get_accuracy_fquad()
