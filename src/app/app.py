import yaml
import pickle
from fastapi import FastAPI
from typing import List
from datetime import datetime
from loguru import logger
from src.app.utils import PostGet, CustomUnpickler
from src.models import Recommender, Recommender_v1


app = FastAPI()
logger.info("Initialized the App!")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    recommender_v1_path = config['recommender_v1_path']
    recommender_v2_path = config['recommender_v2_path']

production_model_path = recommender_v2_path

production_model = CustomUnpickler(open(production_model_path, 'rb')).load()
logger.info("Loaded the Model!")


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    recommendations = production_model.predict(user_id=id)
    return recommendations
