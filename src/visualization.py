# src/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_learning_curves(history):
    """Dibuja las curvas de pérdida y precisión para entrenamiento y validación."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    train_loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    train_acc = history.history.get("accuracy", history.history.get("acc", []))
    val_acc = history.history.get("val_accuracy", history.history.get("val_acc", []))

    ax1.plot(train_loss, label="Entrenamiento")
    if val_loss:
        ax1.plot(val_loss, label="Validación")
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Pérdida")
    ax1.legend()
    ax1.set_title("Curva de Pérdida")

    ax2.plot(train_acc, label="Entrenamiento")
    if val_acc:
        ax2.plot(val_acc, label="Validación")
    ax2.set_xlabel("Épocas")
    ax2.set_ylabel("Precisión")
    ax2.legend()
    ax2.set_title("Curva de Precisión")

    return fig


def plot_decision_boundary(model, X, y, scaler=None, class_names=None,
                           mesh_step=0.02, title="Frontera de Decisión",
                           already_scaled=False):
    """
    Dibuja la frontera de decisión proyectando las características a 2D mediante PCA.
    Acepta datos ya escalados o datos originales junto con un scaler.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if class_names is None:
        class_names = np.arange(len(np.unique(y)))

    if already_scaled:
        X_for_pca = X
    elif scaler is not None:
        X_for_pca = scaler.transform(X)
    else:
        X_for_pca = X

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_for_pca)

    x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
    y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step),
                         np.arange(y_min, y_max, mesh_step))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_model_space = pca.inverse_transform(grid_points)

    pred_proba = model.predict(grid_model_space)
    pred_class = np.argmax(pred_proba, axis=1).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("RdYlBu")
    ax.contourf(xx, yy, pred_class, alpha=0.6, cmap=cmap)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor="k", cmap=cmap)
    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    ax.set_title(title)

    if len(class_names) > 1:
        handles, _ = scatter.legend_elements()
        ax.legend(handles, class_names, title="Clases")

    return fig