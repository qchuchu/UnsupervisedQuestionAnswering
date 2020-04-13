from transformers import CamembertForQuestionAnswering, CamembertTokenizer
from torch import tensor, argmax
import re


class CamembertQuestionAnsweringWrapper:
    def __init__(self, pretrained_model: "str"):
        self.__tokenizer = CamembertTokenizer.from_pretrained(pretrained_model)
        self.__model = CamembertForQuestionAnswering.from_pretrained(pretrained_model)

    def find_answer(self, question: str, context: str):
        input_ids = self.__tokenizer.encode(question, context)
        start_scores, end_scores = self.__model(tensor([input_ids]))
        all_tokens = self.__tokenizer.convert_ids_to_tokens(input_ids)
        answer = "".join(all_tokens[argmax(start_scores) : argmax(end_scores) + 1])
        final_answer = re.sub("▁", " ", answer)
        if final_answer == "" or final_answer.startswith("<s>"):
            return "Not Found"
        return final_answer[0].upper() + final_answer[1:]
