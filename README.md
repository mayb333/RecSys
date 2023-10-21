# RecSys
# About The Project
This repository represents a final project on KarpovCourses StartML - building Post Recommender System for social media and integrating it into a service.


# How it works
In this version of the project, the Content-Based Recommender is implemented using CatBoostClassifier model. 

The data is loaded from the KarpovCourses PostgreSQL DataBase.

The recommendations are being made by passing into the service endpoint a user_id.


# Metrics
The main metric used for this project is `Hitrate@5` (it was chosen by creators of the course)


# Get Started
## Dependencies

* Install the required Python libraries: 
```
pip install -r requirements.txt
```

* Start the app service by running bash-script:
```
bash run_service.sh
```


# Project Structure
1. `static/`: contains script to launch the App service
2. `data/`: contains datasets used in the project (dvc pull)
3. `src/`: contains application source folder
    * `app/`: 
        - `app.py`: source code for the app service
        - `utils.py`: source code for utils for the app service
    * `data/`:
        - `load_data.py`: source code for loading the data
    * `features/`
        - `build_features.py`: source code for feature extraction
    * `metrics/`
        - `metrics.py`: source code for metrics
        - `evaluate_metrics.py`: source code for metrics evaluation
    * `models/`
        - `catboost_recommender_v1`
            + `recommender.py`: source code for Recommender_v1 (baseline)
            + `validation_model_v1.py`: source code for Recommender_v1 validation
            + `artifacts/`: contains saved models
        - `catboost_recommender_v2`
            + `recommender.py`:source code for Recommender_v2 (Production)
            + `validation_model_v2.py`: source code for Recommender_v2 validation
            + `artifacts/`: contains saved models


# The main tools used in the project
![Static Badge](https://img.shields.io/badge/-Python_3.10.8-090909?style=for-the-badge&logo=python&color=black)
![Static Badge](https://img.shields.io/badge/-fastapi_0.75.1-090909?style=for-the-badge&logo=fastapi&color=black)
![Static Badge](https://img.shields.io/badge/-uvicorn_0.16.0-090909?style=for-the-badge&logo=uvicorn&color=black)
![Static Badge](https://img.shields.io/badge/-pandas_1.4.2-090909?style=for-the-badge&logo=pandas&color=black)
![Static Badge](https://img.shields.io/badge/-numpy_1.22.4-090909?style=for-the-badge&logo=numpy&color=black)
![Static Badge](https://img.shields.io/badge/-catboost_1.0.6-090909?style=for-the-badge&logo=catboost&color=black)
![Static Badge](https://img.shields.io/badge/-scikit--learn_1.3.1-090909?style=for-the-badge&logo=scikit-learn&color=black)
![Static Badge](https://img.shields.io/badge/-sqlalchemy_1.4.35-090909?style=for-the-badge&logo=sqlalchemy&color=black)
![Static Badge](https://img.shields.io/badge/-loguru_0.6.0-090909?style=for-the-badge&logo=loguru&color=black)