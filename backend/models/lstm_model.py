"""
lstm_model.py — LSTM model architecture with Attention mechanism.

Architecture (from bottom to top):
  ┌─────────────────────────────────────────────────────┐
  │  Input  (lookback × num_features)                   │
  ├─────────────────────────────────────────────────────┤
  │  LSTM Layer 1  (128 units, return_sequences=True)   │
  │  BatchNormalization                                 │
  │  Dropout (15%)                                      │
  ├─────────────────────────────────────────────────────┤
  │  LSTM Layer 2  (64 units, return_sequences=True)    │
  │  BatchNormalization                                 │
  │  Dropout (15%)                                      │
  ├─────────────────────────────────────────────────────┤
  │  Attention Layer (learns which timesteps matter)     │
  ├─────────────────────────────────────────────────────┤
  │  Dense (32 units, ReLU)                             │
  │  Dropout (15%)                                      │
  ├─────────────────────────────────────────────────────┤
  │  Dense (1 unit) — predicted next-day closing price  │
  └─────────────────────────────────────────────────────┘

Changes from v1 (why these matter for metrics):

  ✓ BatchNormalization after each LSTM
    Stabilizes the internal activations. Without it, the LSTM outputs
    can drift to different scales during training, causing the optimizer
    to oscillate and converge slowly.  This is the single biggest
    improvement for getting R² > 0.9.

  ✓ Huber Loss instead of MSE
    MSE squares the error, so a single outlier day (earnings surprise,
    circuit breaker) can dominate the entire gradient.  Huber loss
    behaves like MSE for small errors and MAE for large ones, making
    training much more stable on financial data.

  ✓ Lower dropout (15% vs 20%)
    With ~300 training sequences, 20% dropout was too aggressive and
    the model couldn't learn enough capacity to generalize.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Layer,
    BatchNormalization,
)
import tensorflow.keras.backend as K

import config


# ────────────────────────────────────────────────────────
#  Custom Attention Layer
# ────────────────────────────────────────────────────────

class AttentionLayer(Layer):
    """
    Bahdanau-style attention over LSTM time steps.

    Given the LSTM output of shape (batch, timesteps, features),
    this layer computes a scalar importance score for each timestep,
    normalizes them with softmax, and returns the weighted sum.

    Steps:
      1. score[t] = tanh(W · h[t] + b)        — project each step
      2. alpha[t] = softmax(score)              — normalize to weights
      3. context  = Σ alpha[t] · h[t]           — weighted combination
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = (batch, timesteps, features)
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="attention_context",
            shape=(input_shape[-1],),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x shape: (batch, timesteps, features)

        # Step 1: Compute scores
        score = K.tanh(K.dot(x, self.W) + self.b)

        # Step 2: Dot with context vector → softmax over timesteps
        attention_weights = K.softmax(K.sum(score * self.u, axis=-1), axis=-1)

        # Step 3: Weighted sum of LSTM outputs
        attention_weights = K.expand_dims(attention_weights, axis=-1)
        context = K.sum(x * attention_weights, axis=1)  # (batch, features)

        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(AttentionLayer, self).get_config()


# ────────────────────────────────────────────────────────
#  Model Builder
# ────────────────────────────────────────────────────────

def build_lstm_model(lookback: int, num_features: int) -> Model:
    """
    Construct and compile the Stacked LSTM + Attention model.

    Parameters
    ----------
    lookback : int
        Number of past timesteps in each input sequence (e.g. 60).
    num_features : int
        Number of input features per timestep (e.g. 11).

    Returns
    -------
    tensorflow.keras.Model
        Compiled model ready for .fit()
    """
    inputs = Input(shape=(lookback, num_features), name="input_sequences")

    # ── LSTM Layer 1: extract low-level temporal patterns ──
    x = LSTM(
        units=config.LSTM_UNITS_LAYER1,
        return_sequences=True,  # pass full sequence to next LSTM
        name="lstm_layer_1",
    )(inputs)
    x = BatchNormalization(name="bn_1")(x)
    x = Dropout(config.DROPOUT_RATE, name="dropout_1")(x)

    # ── LSTM Layer 2: extract higher-order relationships ───
    x = LSTM(
        units=config.LSTM_UNITS_LAYER2,
        return_sequences=True,  # keep sequence for attention
        name="lstm_layer_2",
    )(x)
    x = BatchNormalization(name="bn_2")(x)
    x = Dropout(config.DROPOUT_RATE, name="dropout_2")(x)

    # ── Attention: learn which timesteps matter most ───────
    x = AttentionLayer(name="attention")(x)

    # ── Dense layers for final prediction ──────────────────
    x = Dense(config.DENSE_UNITS, activation="relu", name="dense_hidden")(x)
    x = Dropout(config.DROPOUT_RATE, name="dropout_3")(x)

    outputs = Dense(1, name="price_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="LSTM_Attention_Stock_Predictor")

    # ── Compile with Huber loss ────────────────────────────
    # Huber loss = MSE when |error| < delta, MAE when |error| > delta.
    # This makes training robust to outlier days (earnings surprises,
    # flash crashes) that would distort gradients under pure MSE.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )

    return model
