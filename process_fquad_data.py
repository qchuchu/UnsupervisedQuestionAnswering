from json import load
from typing import List, Tuple
from os import mkdir, walk
from os.path import exists, isdir
from nltk.tokenize import RegexpTokenizer


def load_data(filename: str):

    with open(f"fquad_json_files/{filename}") as json_file:
        json_data = load(json_file)
        articles = json_data["data"]
        titles = list(map(lambda x: x["title"], articles))
        questions = []
        contexts = []
        y_article_context = []
        # This will give the mapping of a context --> article
        context_article_mapping = []
        # We get into an article
        for index_article, article in enumerate(articles):
            # We get into the paragraphs
            context_id = 0
            paragraphs = article["paragraphs"]
            for paragraph in paragraphs:
                qas = paragraph["qas"]
                context = paragraph["context"]
                questions_paragraph = list(map(lambda x: x["question"], qas))
                nb_questions = len(questions_paragraph)
                # We add all the data
                contexts.append(context)
                context_article_mapping.append(index_article)
                questions.extend(questions_paragraph)
                y_article_context.extend(
                    [(index_article, context_id) for _ in range(nb_questions)]
                )
                context_id += 1
        return titles, contexts, questions, context_article_mapping, y_article_context


def store_contexts(
    contexts: List[str],
    context_article_mapping: List[int],
    titles: List[str],
    context_separated: bool,
    tokenizer: RegexpTokenizer,
):
    if context_separated:
        if not isdir("data/fquad_context"):
            mkdir("data/fquad_context")
        context_id = 0
        last_folder = len(next(walk("data/fquad_context"))[1])
        for j in range(len(contexts)):
            context_article_id = context_article_mapping[j]
            title = titles[context_article_id]
            tokenized_title = "_".join(tokenizer.tokenize(title)).lower()
            if not isdir(f"data/fquad_context/{context_article_id + last_folder}"):
                mkdir(f"data/fquad_context/{context_article_id + last_folder}")
                context_id = 0
            with open(
                f"data/fquad_context/{context_article_id + last_folder}/{tokenized_title}_{context_id}.txt",
                "w",
            ) as file:
                file.write(contexts[j])
                context_id += 1
    else:
        if not isdir("data/fquad"):
            mkdir("data/fquad")
            mkdir("data/fquad/0")
        for j in range(len(contexts)):
            context_article_id = context_article_mapping[j]
            title = titles[context_article_id]
            tokenized_title = "_".join(tokenizer.tokenize(title)).lower()
            if exists("data/fquad/0/{}.txt".format(tokenized_title)):
                append_write = "a"
            else:
                append_write = "w"
            with open(
                "data/fquad/0/{}.txt".format(tokenized_title), append_write
            ) as file:
                if append_write == "w":
                    file.write(title + "\n")
                file.write(contexts[j] + "\n")


def store_questions_answers(
    questions: List[str],
    titles: List[str],
    y_article_context: List[Tuple[int, int]],
    tokenizer: RegexpTokenizer,
    filename: str,
):
    for index, question in enumerate(questions):
        with open(f"dev_resources/queries/{filename.split('.')[0]}.in", "a") as file:
            file.write(question + "\n")
        with open(
            f"dev_resources/output/{filename.split('.')[0]}.out".format(index), "a"
        ) as file:
            article_id, context_id = y_article_context[index]
            title = titles[article_id]
            tokenized_title = "_".join(tokenizer.tokenize(title)).lower()
            title_context = "{}_{}".format(tokenized_title, context_id)
            file.write(title_context + "\n")


def extract_data_from_fquad(filename: str):
    """
    This function extracts the data from the Fquad Train dataset and store the queries, the expected context
    associated (in dev_resources) and the content of the context (in data/fquad)
    :return:
    """
    tokenizer = RegexpTokenizer(r"\w+")
    titles, contexts, questions, context_article_mapping, y_article_context = load_data(
        filename
    )
    store_contexts(contexts, context_article_mapping, titles, True, tokenizer)
    store_contexts(contexts, context_article_mapping, titles, False, tokenizer)
    store_questions_answers(questions, titles, y_article_context, tokenizer, filename)


if __name__ == "__main__":
    extract_data_from_fquad("fquad_train.json")
    extract_data_from_fquad("fquad_valid.json")
