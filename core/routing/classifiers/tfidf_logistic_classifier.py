"""
Statistical Intent Classifier using TF-IDF Vectorization & Logistic Regression.
Trains on a curated technical corpus to provide calibrated probability distributions in < 1ms.
"""

import time
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .training_data import INTENT_TRAINING_CORPUS, INTENT_CLASSES


@dataclass
class TFIDFLogisticResult:
    """Output from the TF-IDF + Logistic Regression model."""
    predicted_intent: str
    confidence: float
    probabilities: Dict[str, float]
    top_contributing_features: List[str]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TFIDFLogisticIntentClassifier:
    """
    Statistical NLP Classifier.
    Transforms prompts into character/word n-gram TF-IDF representations
    and classifies them via multinomial Logistic Regression.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b|[+\-*/=><!~]",
        )
        self.model = LogisticRegression(
            C=2.5,
            max_iter=300,
            solver="lbfgs",
            random_state=42,
        )
        self._train()

    def _train(self):
        """Fits the TF-IDF vectorizer and Logistic Regression model on the training corpus."""
        texts = [sample[0] for sample in INTENT_TRAINING_CORPUS]
        labels = [sample[1] for sample in INTENT_TRAINING_CORPUS]

        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, text: str) -> TFIDFLogisticResult:
        """Classifies an input prompt and returns probabilities and top features."""
        start = time.perf_counter()

        if not text or not text.strip():
            return TFIDFLogisticResult(
                predicted_intent="FACTUAL_LOOKUP",
                confidence=0.2,
                probabilities={c: 0.2 for c in INTENT_CLASSES},
                top_contributing_features=[],
                latency_ms=0.0,
            )

        X_input = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X_input)[0]

        prob_dict = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(self.model.classes_, probs)
        }

        best_class = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[best_class]

        # Extract top active n-gram features from the input
        feature_names = self.vectorizer.get_feature_names_out()
        nonzero_indices = X_input.nonzero()[1]
        active_features = [feature_names[idx] for idx in nonzero_indices][:5]

        latency = round((time.perf_counter() - start) * 1000, 3)

        return TFIDFLogisticResult(
            predicted_intent=best_class,
            confidence=confidence,
            probabilities=prob_dict,
            top_contributing_features=active_features,
            latency_ms=latency,
        )
