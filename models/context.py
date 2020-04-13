from utils.text_transformer import TextTransformer
from torch import tensor


class Context:
    def __init__(self, content: str):
        self.content = content
        self.embedding: tensor = tensor([])

    def encode(self, text_transformer: TextTransformer):
        self.embedding = text_transformer.encode(self.content)


if __name__ == "__main__":
    text_transformer = TextTransformer()
    content_query = (
        "Quelles furent les découvertes finales des vingt-quatre astronomes ?"
    )
    context = Context(content_query)
    context.encode(text_transformer)
    print(context.embedding)
