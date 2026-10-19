import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def synthetic_table():
    """Software fixture only; never written into research submissions."""
    features, target = make_classification(
        n_samples=60, n_features=288, n_informative=8,
        n_redundant=4, random_state=42,
    )
    groups = np.repeat(np.arange(60), 2)
    features = np.repeat(features, 2, axis=0)
    # An audit marker used to prove which patients each scaler was fitted on.
    features[:, 0] = groups
    table = pd.DataFrame(features, columns=[f"original_{i}" for i in range(288)])
    table["image_id"] = [f"synthetic-image-{i}" for i in range(120)]
    table["patient_id"] = groups
    table["source_case"] = "synthetic"
    table["diagnosis"] = np.repeat(target, 2)
    table["pathology"] = "synthetic"
    table["fold5_kfold"] = groups % 5 + 1
    table["fold10_kfold"] = groups % 10 + 1
    return table
