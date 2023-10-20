import pickle
import yaml
import re
import pandas as pd
import numpy as np
from src.models.catboost_recommender_v1 import Recommender_v1_validation
from src.metrics import hitrate_at_k


def evaluate_metrics(model, model_name: str = 'Recommender_v1'):
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        validation_path = config['validation_data_path']

    validation = pd.read_csv(validation_path, sep=';', index_col=None)

    hitrate = 0
    for i in range(len(validation)):
        user_id = validation.user_id.iloc[i]
        liked_posts = []
        viewed_posts = []

        for post_str in re.sub(pattern=r'(\[|\])', repl='', string=validation.liked_posts.iloc[i]).split(','):
            post_id = int(post_str.strip())
            liked_posts.append(post_id)

        for post_str in re.sub(pattern=r'(\[|\])', repl='', string=validation.viewed_posts.iloc[i]).split(','):
            post_id = int(post_str.strip())
            viewed_posts.append(post_id)

        viewed_posts = np.array(viewed_posts)    

        recs = [item['id'] for item in model.predict(user_id=user_id, viewed_posts=viewed_posts)]
        hitrate += hitrate_at_k(liked_posts=liked_posts, recommended_posts=recs)

    hitrate_at_5 = round(hitrate / len(validation), 3)

    print(f"Model: {model_name}")
    print(f"Hitrate@5 = {hitrate_at_5}")

    return hitrate_at_5


if __name__ == '__main__':
    model_v1 = pickle.load(open('src/models/catboost_recommender_v1/validation_model_v1.pkl', 'rb'))
    evaluate_metrics(model=model_v1, model_name='Recommender_v1') #hitrate@5 = 0.572
