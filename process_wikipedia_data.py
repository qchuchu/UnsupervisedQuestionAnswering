from pickle import load
from os import mkdir
from os.path import isdir
from nltk.tokenize import RegexpTokenizer


def load_data():
    tokenizer = RegexpTokenizer(r"\w+")
    with open('wikipedia_best_quality_article/contexts_1.p', 'rb') as pickle_file:
        articles = load(pickle_file)
    with open('wikipedia_best_quality_article/contexts_2.p', 'rb') as pickle_file:
        articles.extend(load(pickle_file))

    if not isdir('data/wiki'):
        mkdir('data/wiki')
    if not isdir('data/wiki_context'):
        mkdir('data/wiki_context')
    for index_article, article in enumerate(articles):
        title = article['title']
        tokenized_title = "_".join(tokenizer.tokenize(title)).lower()
        contexts = article['contexts']
        article_directory_path = f'data/wiki_context/{index_article}'
        article_path = f'data/wiki/{tokenized_title}.txt'
        article_file = open(article_path, 'w')
        article_file.write(title)
        if not isdir(article_directory_path):
            mkdir(article_directory_path)
        for index_context, context in enumerate(contexts):
            context_path = f'{article_directory_path}/{tokenized_title}_{index_context}.txt'
            article_file.write(f'\n{context}')
            with open(context_path, 'w') as context_file:
                context_file.write(context)


if __name__ == '__main__':
    load_data()
