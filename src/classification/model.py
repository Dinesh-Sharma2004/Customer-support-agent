from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os
from src.config.settings import settings

def build_base_classifier() -> Pipeline:
    """Builds the uncalibrated champion pipeline."""
    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=80_000,
        sublinear_tf=True,
        min_df=2
    )

    char_tfidf = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=50_000,
        sublinear_tf=True,
        min_df=5
    )

    features = FeatureUnion([
        ('word', word_tfidf),
        ('char', char_tfidf)
    ])

    clf_pipeline = Pipeline([
        ('features', features),
        ('clf', LogisticRegression(
            C=5,
            max_iter=1000,
            class_weight='balanced',
            random_state=settings.RANDOM_SEED,
            solver='lbfgs',
            n_jobs=-1
        ))
    ])
    
    return clf_pipeline

from sklearn.frozen import FrozenEstimator

def build_calibrated_classifier(base_pipeline: Pipeline) -> CalibratedClassifierCV:
    """Wraps a fitted base pipeline in a CalibratedClassifierCV using sigmoid."""
    return CalibratedClassifierCV(
        estimator=FrozenEstimator(base_pipeline),
        method='sigmoid'
    )

def save_model(model, filename: str):
    path = os.path.join(settings.ARTIFACTS_DIR, filename)
    joblib.dump(model, path)
    return path

def load_model(filename: str):
    path = os.path.join(settings.ARTIFACTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)
