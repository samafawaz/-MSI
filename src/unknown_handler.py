import numpy as np
from .constants import UNKNOWN_CLASS_ID


class SVMUnknownHandler:
    """
    UNKNOWN detection using:
      - Softmax probability
      - Probability margin
    """

    def __init__(self, svm_pipeline, threshold=0.65, margin=0.08  ):
        self.svm_pipeline = svm_pipeline
        self.threshold = threshold
        self.margin = margin

    def _scores_to_probs(self, scores):
        scores -= np.max(scores)
        exps = np.exp(scores)
        return exps / (np.sum(exps) + 1e-8)

    def predict_single(self, x):
        x = np.asarray(x).reshape(1, -1)

        decision_scores = self.svm_pipeline.decision_function(x)[0]

        probs = self._scores_to_probs(decision_scores)

        best_label = int(np.argmax(probs))
        top1 = float(probs[best_label])
        top2 = float(np.sort(probs)[-2])

        margin = top1 - top2

        # UNKNOWN logic:
        if top1 < self.threshold:
            return UNKNOWN_CLASS_ID

        if margin < self.margin:
            return UNKNOWN_CLASS_ID

        return best_label
