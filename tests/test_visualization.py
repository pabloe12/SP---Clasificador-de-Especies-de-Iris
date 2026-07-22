import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.visualization import plot_decision_boundary


class DummyModel:
    def predict(self, X):
        return np.tile(np.array([0.2, 0.3, 0.5]), (len(X), 1))


def test_plot_decision_boundary_accepts_scaled_inputs():
    X = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [6.0, 2.2, 4.0, 1.0],
        [6.3, 2.8, 5.1, 1.5],
    ])
    y = np.array([0, 0, 1, 1])
    class_names = np.array(["setosa", "versicolor"])

    fig = plot_decision_boundary(
        DummyModel(),
        X,
        y,
        scaler=None,
        class_names=class_names,
        already_scaled=True,
    )

    assert fig is not None
