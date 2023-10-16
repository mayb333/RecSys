import pandas as pd
import numpy as np
from typing import List


def hitrate_at_k(liked_posts: List[int], recommended_posts: List[int], k: int = 5) -> float:
    liked_posts = np.array(liked_posts)
    recommended_posts = np.array(recommended_posts)

    flags = np.isin(recommended_posts[:k], liked_posts)
    hitrate = int(flags.sum() > 0)

    return hitrate
