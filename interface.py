import time

import click
import pyfiglet
from pickle import load

from models.search_engine import SearchEngine
from utils.text_transformer import TextTransformer
from utils.camembert_question_answering_wrapper import CamembertQuestionAnsweringWrapper


@click.command()
@click.option("-c", "--count", default=10, help="Number of results.", type=int)
@click.option(
    "-n", "--collection_name", default="wiki_context", help="Collection Name", type=str
)
@click.option(
    "-w",
    "--weighting_model",
    default="okapi-bm25",
    type=click.Choice(["tw-idf", "tf-idf", "okapi-bm25"], case_sensitive=False),
    help="Weighting Type.",
)
@click.option(
    "-m",
    "--embedding_model",
    default="camembert-fquad",
    help="Context Embedding Model",
    type=str,
)
@click.option("-l", "--lemmatizer", default="spacy-fr", help="Lemmatizer used")
@click.option(
    "-M",
    "--mean_tokens",
    default=False,
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
    default=False,
    help="If you want to make a deeper search on the contexts inside a document or not",
    type=bool,
)
@click.option(
    "-q",
    "--question_answering_label",
    default="./camembert_fine_tuned_13000_questions",
    help="Question Answering"
)
def interface(
    count,
    collection_name,
    weighting_model,
    embedding_model,
    lemmatizer,
    mean_tokens,
    cls_token,
    max_tokens,
    document_tokenizer,
    context_retrieval,
    question_answering_label
):
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
    question_answering_model = CamembertQuestionAnsweringWrapper(
        question_answering_label
    )
    click.clear()

    def search(query):
        """
        This function is the main interface for querying the search engine.
        """
        start_time = time.time()
        click.secho("Searching query ...", fg="blue", bold=True)
        scores_query = search_engine.search(query)

        click.secho("Sorting results ...", fg="blue", bold=True)
        sorted_results = [
            k
            for k, v in sorted(
                scores_query.items(), key=lambda item: item[1], reverse=True
            )
        ]
        finished_time = time.time()
        total_time = round((finished_time - start_time) * 1000, 2)
        click.secho(
            "Finished ! Total time: {}ms".format(total_time), fg="green", bold=True
        )
        total_results = min(count, len(sorted_results))
        if text_transformer.context_retrieval:
            for i, (doc_id_query, context_name) in enumerate(
                sorted_results[:total_results]
            ):
                document = search_engine.collection.documents[doc_id_query]
                context_filename = "contexts/{}_{}_{}.p".format(
                    text_transformer.embedding_model_label,
                    text_transformer.pooling_modes,
                    context_name,
                )
                with open(context_filename, "rb") as pickle_context:
                    context = load(pickle_context)
                click.secho("{}.\t{}".format(i, context_name), bold=True)
                click.secho("\t{}\n".format(" ".join(document.key_words)), fg="red")
                click.secho(context.content, fg="green")
        else:
            for i, doc_id_query in enumerate(sorted_results[:total_results]):
                document = search_engine.collection.documents[doc_id_query]
                click.secho(
                    "{}.\t{}/{}".format(i, document.folder, document.url.split(".")[0]),
                    bold=True,
                )
                click.secho("\t{}\n".format(document.content), fg="green")
                click.secho(
                    "Possible Answer :{}".format(
                        question_answering_model.find_answer(query, document.content)
                    ),
                    bold=True,
                    fg="red",
                )

    while True:
        result = pyfiglet.figlet_format("Alexa 4.0", font="big")
        click.secho(result, fg="red", bold=True)
        user_query = click.prompt(
            click.style("Please enter you query", fg="blue", bold=True), type=str
        )
        search(user_query)
        click.confirm("Do you want to continue?", abort=True)
        click.clear()


if __name__ == "__main__":
    interface()
