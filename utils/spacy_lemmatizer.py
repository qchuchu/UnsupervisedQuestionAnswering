import spacy


class SpacyLemmatizer:
    def __init__(self):
        print("Loading Spacy NLP French Model : fr_core_news_md")
        self.__nlp = spacy.load("fr_core_news_md")
        print("Finish Loading")

    def lemmatize(self, sentence):
        tokenized_sentence = self.__nlp(sentence)
        return [token.lemma_ for token in tokenized_sentence]


if __name__ == "__main__":
    spacy_french_lemmatizer = SpacyLemmatizer()
    sentence = "le chien est beau"
    print(spacy_french_lemmatizer.lemmatize(sentence))
