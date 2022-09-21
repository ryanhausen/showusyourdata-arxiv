from turtle import st
from typing import List


class Model:
    def preprocess(self, json_text: str) -> str:
        raise NotImplementedError("Model doesn't implement preprocess")

    def predict(self, text: str) -> List[str]:
        raise NotImplementedError("Model doesn't implement predict")
