import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, ReLU, Activation, Embedding
from tensorflow_addons.layers import WeightNormalization

class DeepSDFDecoder(keras.Model):
    def __init__(self, latent_dim, num_shapes, shape_code_dim, hidden_dim=512, dropout_rate=0.2):
        super(DeepSDFDecoder, self).__init__()

        # store shape codes in trainable embedding layer 
        self.latent_shape_code_emb = Embedding(num_shapes, shape_code_dim, embeddings_initializer='normal')

        # Head: 4 FC layers [B, (latent_dim+3)] -> [B, (hidden-(latent_dim+3))]
        self.head = keras.Sequential([
            WeightNormalization(Dense(hidden_dim)),
            Dropout(dropout_rate),
            ReLU(),
            WeightNormalization(Dense(hidden_dim)),
            Dropout(dropout_rate),
            ReLU(),
            WeightNormalization(Dense(hidden_dim)),
            Dropout(dropout_rate),
            ReLU(),
            WeightNormalization(Dense(hidden_dim-(latent_dim+3))),
            Dropout(dropout_rate),
            ReLU(),
        ])

        # Tail: 4 FC layers + tanh output activation [B, 512] -> [B,1]
        self.tail = keras.Sequential([
            WeightNormalization(Dense(hidden_dim)),
            Dropout(dropout_rate),
            ReLU(),
            WeightNormalization(Dense(hidden_dim)),
            Dropout(dropout_rate),
            ReLU(),
            WeightNormalization(Dense(hidden_dim)),
            Dropout(dropout_rate),
            ReLU(),
            WeightNormalization(Dense(1)),
            Activation('tanh')
        ])

    def call(self, x, shape_idx, training=False):
        shape_code = self.latent_shape_code_emb(shape_idx)
        input = tf.concat([shape_code, x], axis=1) # [B, (latent_dim+3)]
        intermediate = self.head(input) # [B, hidden-(latent_dim+3)]
        # skip connection
        intermediate = tf.concat([intermediate, input], axis=1) # [B, hidden_dim]
        return self.tail(intermediate)

    def loss(self, sdf_pred, sdf_true):
        return keras.losses.MeanAbsoluteError()(sdf_true, sdf_pred)

