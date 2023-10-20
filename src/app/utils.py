import pickle
import datetime
from pydantic import BaseModel
from typing import List


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'Recommender_v1':
            from src.models import Recommender_v1
            return Recommender_v1
        elif name == 'Recommender':
            from src.models import Recommender
            return Recommender
        return super().find_class(module, name)