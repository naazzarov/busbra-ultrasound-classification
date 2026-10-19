"""Independent paper-based selectors; fitted only on the current training data.

Definitions, source discrepancies and fixed settings: docs/METHODS.md.
No author implementation is redistributed or imported.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class _Selector(TransformerMixin, BaseEstimator):
    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Predictor count differs from selector training")
        return X[:, self.selected_indices_]

    def get_support(self, indices=False):
        check_is_fitted(self, "selected_indices_")
        if indices:
            return self.selected_indices_.copy()
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_indices_] = True
        return mask


class FuzzyRoughSelector(_Selector):
    """Jensen & Shen (2009), similarity Eq. 26, Lukasiewicz operators.

    The crisp binary decision simplifies the positive region to the minimum
    dissimilarity to an opposite-class object. Dissimilarities combine by a
    capped sum (the dual of the Lukasiewicz t-norm), not a Euclidean norm.
    """
    def __init__(self, tol=1e-12):
        self.tol = tol

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float)
        if set(y) != {0, 1}:
            raise ValueError("Fuzzy-rough selection requires both binary classes")
        self.n_features_in_ = X.shape[1]
        sigma = X.std(axis=0, ddof=1)
        sigma[sigma == 0] = 1
        negative, positive = X[y == 0] / sigma, X[y == 1] / sigma
        accumulated = np.zeros((len(negative), len(positive)))
        selected, remaining, history = [], list(range(X.shape[1])), []
        dependency = 0.0
        while remaining:
            best_score, best_feature = dependency, None
            for feature in remaining:
                distance = np.abs(negative[:, feature, None] - positive[None, :, feature])
                candidate = np.minimum(accumulated + np.minimum(distance, 1), 1)
                score = float((candidate.min(axis=1).sum() + candidate.min(axis=0).sum()) / len(X))
                if score > best_score + self.tol:
                    best_score, best_feature = score, feature
            if best_feature is None:
                break
            accumulated = np.minimum(accumulated + np.minimum(
                np.abs(negative[:, best_feature, None] - positive[None, :, best_feature]), 1), 1)
            selected.append(best_feature)
            remaining.remove(best_feature)
            dependency = best_score
            history.append(best_score)
            if dependency >= 1 - self.tol:
                break
        if not selected:
            raise ValueError("No feature improves fuzzy dependency; no arbitrary fallback is permitted")
        self.selected_indices_ = np.asarray(selected, dtype=int)
        self.dependency_history_ = history
        return self


@dataclass
class Ball:
    indices: np.ndarray
    center: np.ndarray
    radius: float
    label: int
    purity: float


def _ball(X, y, indices):
    center = X[indices].mean(axis=0)
    labels, counts = np.unique(y[indices], return_counts=True)
    # The author uses first occurrence on majority ties; retain that convention.
    tied = set(labels[counts == counts.max()])
    label = next(int(label) for label in y[indices] if label in tied)
    return Ball(indices, center, float(np.linalg.norm(X[indices] - center, axis=1).mean()),
                label, float(counts.max() / len(indices)))


def build_balls(X, y, *, purity=0.9, random_state=42):
    """Class-stratified sqrt-count seeds, purity splitting, overlap removal.

    Unlike the released script, explicitly fail on irreducible geometry rather
    than creating empty balls or silently leaving unresolved overlaps.
    """
    rng = np.random.RandomState(random_state)
    seeds = np.concatenate([rng.choice(np.flatnonzero(y == label),
                            int(np.sqrt(np.sum(y == label))), replace=False)
                            for label in np.unique(y)])
    labels = KMeans(n_clusters=len(seeds), init=X[seeds], n_init=1,
                    random_state=random_state).fit_predict(X)
    balls = [_ball(X, y, np.flatnonzero(labels == label)) for label in np.unique(labels)]

    def split(ball, supervised):
        if len(ball.indices) < 2:
            return [ball]
        values = X[ball.indices]
        if supervised:
            local_y = y[ball.indices]
            sums = values.sum(axis=1)
            init = np.asarray([values[np.flatnonzero(local_y == label)[
                np.argmin(sums[local_y == label])]] for label in (0, 1)])
            assignments = KMeans(n_clusters=2, init=init, n_init=1,
                                  random_state=random_state).fit_predict(values)
        else:
            assignments = KMeans(n_clusters=2, random_state=5, n_init=1).fit_predict(values)
        return [_ball(X, y, ball.indices[assignments == label]) for label in np.unique(assignments)]

    pending, balls = balls, []
    while pending:
        ball = pending.pop(0)
        if ball.purity >= purity:
            balls.append(ball)
        else:
            children = split(ball, True)
            if len(children) < 2:
                raise ValueError("GBFRS cannot split an impure ball with coincident features")
            pending.extend(children)
    # Recheck globally after every pass; all samples remain in exactly one ball.
    for _ in range(len(X)):
        centers = np.asarray([b.center for b in balls])
        radii = np.asarray([b.radius for b in balls])
        classes = np.asarray([b.label for b in balls])
        overlap = (cdist(centers, centers) < radii[:, None] + radii[None, :]) & (classes[:, None] != classes[None, :])
        affected = overlap.any(axis=1)
        if not affected.any():
            if set(classes) != {0, 1}:
                raise ValueError("GBFRS ball labels lost a class; raise purity before a new declared experiment")
            return balls
        revised, progressed = [], False
        for ball, needs_split in zip(balls, affected):
            children = split(ball, False) if needs_split else [ball]
            progressed |= len(children) > 1
            revised.extend(children)
        if not progressed:
            raise ValueError("GBFRS heterogeneous overlaps cannot be resolved")
        balls = revised
    raise ValueError("GBFRS overlap removal did not converge")


def ball_dependency(centers, labels, weights, indices, normalizer):
    """Xia et al. Eq. 13: own-ball weights times nearest heterogeneous distance.

    The caller fixes a valid distance normalizer, or uses |B| on the paper's
    [0,1] scale. No clipping that would change the paper's ranking is applied.
    """
    distances = cdist(centers[:, indices], centers[:, indices])
    distances[labels[:, None] == labels[None, :]] = np.inf
    positive_region = distances.min(axis=1) / np.sqrt(normalizer)
    return float(weights @ positive_region)


class GBFRSSelector(_Selector):
    """Paper-defined GBFRS, fixed ball memberships and weighted dependency.

    Standard-scaled training values are mapped to [0,1] only for ball geometry,
    matching the paper's domain. SVM receives the original standardized columns.
    """
    def __init__(self, purity=0.9, random_state=42, tol=1e-12):
        self.purity = purity
        self.random_state = random_state
        self.tol = tol

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float)
        if set(y) != {0, 1} or not 0.5 < self.purity <= 1:
            raise ValueError("GBFRS requires both binary classes and purity in (0.5,1]")
        self.n_features_in_ = X.shape[1]
        span = np.ptp(X, axis=0)
        span[span == 0] = 1
        geometry = (X - X.min(axis=0)) / span
        balls = build_balls(geometry, y, purity=self.purity, random_state=self.random_state)
        self.ball_sizes_ = np.asarray([len(ball.indices) for ball in balls])
        self.ball_labels_ = np.asarray([ball.label for ball in balls])
        centers = np.asarray([ball.center for ball in balls])
        weights = self.ball_sizes_ / len(X)
        selected, remaining, history = [], list(range(X.shape[1])), []
        dependency = 0.0
        while remaining:
            scores = [ball_dependency(centers, self.ball_labels_, weights,
                      selected + [feature], len(selected) + 1) for feature in remaining]
            best = int(np.argmax(scores))
            # The preprint Algorithm 1 prose and released code stop when the
            # normalized dependency no longer improves over the previous set.
            if scores[best] <= dependency + self.tol:
                break
            selected.append(remaining.pop(best))
            dependency = scores[best]
            history.append(dependency)
        if not selected:
            raise ValueError("No feature improves GBFRS dependency")
        self.selected_indices_ = np.asarray(selected, dtype=int)
        self.dependency_history_ = history
        return self


def make_selector(method):
    if method == "baseline":
        return None
    if method == "fuzzy_rough":
        return FuzzyRoughSelector()
    if method == "gbfrs":
        return GBFRSSelector()
    raise ValueError(f"Unknown method: {method}")
