from dspy import Example
import json
from typing import Literal
import random

class Data:
    def __init__(self, data: list[Example], split: Literal["base", "train", "dev", "test"] = "base"):
        self.data = data
        self.split = split
        self.scores = [-1.0]*self.length

    @property
    def length(self):
        return len(self.data)
    
    def get_splits(self, n: Literal[2, 3]):
        if self.split != "base":
            raise ValueError(f"Can't get splits from '{self.split}'")
        ss = len(self.data) // 3
        if n == 2:
            train, test = self.data[:2*ss], self.data[2*ss:]
            return Data(train, split = "train"), Data(test, split = "test")
        else:
            train, dev, test = self.data[:ss], self.data[ss:2*ss], self.data[2*ss:]
            return Data(train, split = "train"), Data(dev, split = "dev"), Data(test, split = "test")

    @classmethod
    def from_json(cls, path):
        with open(f"{path}", "r") as f:
            data = json.load(f)
        def examplify(s):
            ret = Example(question=s["question"], answer=s["answer"])
            ret.with_inputs("question")
            return ret
        
        data = map(examplify, data)
        return cls(list(data))

    def __str__(self):
        TEMPLATE = "Question: {q}\nAnswer: {a}"
        example_strings = [TEMPLATE.format(q=e.question, a=e.answer) for e in self.data]
        return "\n".join(example_strings)
    


