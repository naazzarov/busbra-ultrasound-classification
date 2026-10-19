"""The immutable protocol shared by all experimental methods."""
import json
from importlib.resources import files


def read_contract() -> dict:
    return json.loads(files("busbra").joinpath("resources/evaluation_contract.json").read_text())


def svm_grid() -> list[dict]:
    grid = read_contract()["common_svm_grid"]
    common = {"svm__C": grid["C"], "svm__class_weight": grid["class_weight"]}
    return [
        dict(common, svm__kernel=["linear"]),
        dict(common, svm__kernel=["rbf"], svm__gamma=grid["rbf_gamma"]),
    ]
