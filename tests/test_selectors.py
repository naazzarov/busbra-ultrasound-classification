import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from busbra.selectors import FuzzyRoughSelector, GBFRSSelector, ball_dependency, build_balls


def test_jensen_shen_six_object_example():
    X = np.array([[-.4,-.3,-.5],[-.4,.2,-.1],[-.3,-.4,-.3],[.3,-.3,0],[.2,-.3,0],[.2,0,0]])
    y = np.array([0,1,0,1,1,0])
    selector = FuzzyRoughSelector().fit(X, y)
    assert selector.get_support(indices=True).tolist() == [2,1]
    assert selector.dependency_history_[0] == pytest.approx(.4078, abs=.001)
    assert selector.dependency_history_[-1] == 1
    np.testing.assert_allclose(selector.transform(X), X[:, [2,1]])
    scaled = FuzzyRoughSelector().fit(StandardScaler().fit_transform(X), y)
    assert scaled.get_support(indices=True).tolist() == [2,1]


def test_paper_dependency_weights_own_ball_not_neighbor():
    centers = np.array([[0.,0.],[.2,.8],[.9,.9]])
    labels = np.array([0,0,1])
    weights = np.array([.6,.3,.1])
    # Nearest heterogeneous distances for the first column: .9,.7,.7.
    assert ball_dependency(centers, labels, weights, [0], 1) == pytest.approx(.82)
    # Independent direct implementation of Eq 13 on two coordinates.
    expected = .6*np.hypot(.9,.9) + .4*np.hypot(.7,.1)
    assert ball_dependency(centers, labels, weights, [0,1], 2) == pytest.approx(expected/np.sqrt(2))


def test_gbfrs_geometry_coverage_and_determinism():
    rng = np.random.RandomState(10)
    X = np.r_[rng.normal(.1,.01,(20,3)),rng.normal(.8,.01,(20,3))]
    y = np.repeat([0,1],20)
    with threadpool_limits(limits=1):
        balls = build_balls(X, y)
        a = GBFRSSelector().fit(X, y)
        b = GBFRSSelector().fit(X, y)
    indices = np.concatenate([ball.indices for ball in balls])
    assert sorted(indices) == list(range(40))
    assert all(ball.purity >= .9 for ball in balls)
    for first in balls:
        for second in balls:
            if first.label != second.label:
                assert np.linalg.norm(first.center-second.center) >= first.radius+second.radius
    assert a.get_support(indices=True).tolist() == b.get_support(indices=True).tolist()
    assert a.ball_sizes_.sum() == len(X)
    assert 1 <= len(a.selected_indices_) <= X.shape[1]
    assert np.all(np.diff(a.dependency_history_) > 0)
    assert all(0 <= d <= 1 for d in a.dependency_history_)


@pytest.mark.parametrize("selector", [FuzzyRoughSelector(), GBFRSSelector()])
def test_invalid_training_fails_instead_of_fabricating_subset(selector):
    with pytest.raises(ValueError, match="binary classes"):
        selector.fit([[0],[1]], [0,0])
