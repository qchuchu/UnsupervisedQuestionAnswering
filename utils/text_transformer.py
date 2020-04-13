from sentence_transformers import SentenceTransformer
from sentence_transformers.models import CamemBERT, Pooling
from nltk.tokenize import RegexpTokenizer
from utils.spacy_lemmatizer import SpacyLemmatizer
from torch import tensor

LEMMATIZERS = {"spacy-fr": SpacyLemmatizer}

MODELS = {
    "camembert-base": CamemBERT,
    "fmikaelian/camembert-base-fquad": CamemBERT,
    "./camembert-fine-tuned": CamemBERT,
}

TOKENIZERS = {"only-words": r"\w+"}

CAMEMBERT_LABEL_TRANSLATOR = {
    "camembert-base": "camembert-base",
    "camembert-fquad": "fmikaelian/camembert-base-fquad",
    "camembert-fine-tuned": "./camembert-fine-tuned",
}


class TextTransformer:
    def __init__(
        self,
        lemmatizer_label: str = "spacy-fr",
        embedding_model_label: str = "camembert-base",
        document_tokenizer: str = "only-words",
        mean_tokens: bool = True,
        cls_token: bool = True,
        max_tokens: bool = False,
        context_retrieval: bool = True,
    ):
        self.__lemmatizer_label = lemmatizer_label
        if context_retrieval:
            self.__embedding_model_label = embedding_model_label
            print(f"Loading {embedding_model_label} model...")
            self.__embedding_model = MODELS[
                CAMEMBERT_LABEL_TRANSLATOR[embedding_model_label]
            ](CAMEMBERT_LABEL_TRANSLATOR[embedding_model_label])
            print(f"Finished Loading {embedding_model_label} model !")
            print(
                "Creating Pooling Model..\nMean Tokens : {}\nCLS Token : {}\nMax Tokens : {}".format(
                    mean_tokens, cls_token, max_tokens
                )
            )
            self.__pooling_model = Pooling(
                self.__embedding_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=mean_tokens,
                pooling_mode_cls_token=cls_token,
                pooling_mode_max_tokens=max_tokens,
            )
            print("Pooling Model Created !")
            self.__sentence_transformer = SentenceTransformer(
                modules=[self.__embedding_model, self.__pooling_model]
            )
            self.__pooling_modes: str = ""
            modes = []
            if mean_tokens:
                modes.append("mean")
            if cls_token:
                modes.append("cls")
            if max_tokens:
                modes.append("max")
            self.__pooling_modes = "_".join(modes)
        else:
            self.__embedding_model_label = ""
            self.__sentence_transformer = None
            self.__pooling_model = None
            self.__pooling_modes = ""
        self.__lemmatizer = LEMMATIZERS[lemmatizer_label]()
        self.__document_tokenizer = RegexpTokenizer(TOKENIZERS[document_tokenizer])
        self.context_retrieval = context_retrieval

    @property
    def pooling_modes(self):
        return self.__pooling_modes

    @property
    def lemmatizer_label(self):
        return self.__lemmatizer_label

    @property
    def embedding_model_label(self):
        return self.__embedding_model_label

    @property
    def lemmatizer(self):
        return self.__lemmatizer

    def lemmatize(self, sentence: str):
        return self.__lemmatizer.lemmatize(sentence)

    def encode(self, sentence: str):
        return tensor(self.__sentence_transformer.encode([sentence])[0])

    def tokenize(self, sentence: str, lower_case: bool = True):
        tokenized_sentence = self.__document_tokenizer.tokenize(sentence)
        if lower_case:
            tokens = []
            for token in tokenized_sentence:
                tokens.append(token.lower())
            return tokens
        else:
            return tokenized_sentence


if __name__ == "__main__":
    text_transformer = TextTransformer()
