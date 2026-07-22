# src/train.py
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def build_model(input_dim=4, hidden1=8, hidden2=6, output_dim=3):
    """Construye la MLP con las capas especificadas."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(hidden1, activation='relu'),
        Dense(hidden2, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(X_train, y_train, X_val, y_val,
                epochs=200, batch_size=16, patience=20):
    """
    Entrena el modelo con early stopping.
    y_train, y_val deben estar en formato one-hot (categorical).
    """
    model = build_model()
    early_stop = EarlyStopping(monitor='val_loss', patience=patience,
                               restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    return model, history