import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, ReLU, Activation, Embedding
from tensorflow.keras.initializers import RandomNormal
from tensorflow_addons.layers import WeightNormalization

class DeepSDFDecoder(keras.Model):
    def __init__(self, num_shapes, shape_code_dim, hidden_dim=512, dropout_rate=0.2):
        super(DeepSDFDecoder, self).__init__()

        # store shape codes in trainable embedding layer 
        emb_init = RandomNormal(mean=0.0, stddev=0.01, seed=None)
        self.latent_shape_code_emb = Embedding(num_shapes, shape_code_dim, embeddings_initializer=emb_init)

        # Head: 4 FC layers [B, (shape_code_dim+3)] -> [B, (hidden-(shape_code_dim+3))]
        # TODO: try WeightNorm
        self.head = keras.Sequential([
            Dense(hidden_dim),
            Dropout(dropout_rate),
            ReLU(name='head_relu_1'),
            Dense(hidden_dim),
            Dropout(dropout_rate),
            ReLU(name='head_relu_2'),
            Dense(hidden_dim),
            Dropout(dropout_rate),
            ReLU(name='head_relu_3'),
            Dense(hidden_dim-(shape_code_dim+3)),
            Dropout(dropout_rate),
            ReLU(name='head_relu_4'),
        ])

        # Tail: 4 FC layers + tanh output activation [B, 512] -> [B,1]
        self.tail = keras.Sequential([
            Dense(hidden_dim),
            Dropout(dropout_rate),
            ReLU(name='tail_relu_1'),
            Dense(hidden_dim),
            Dropout(dropout_rate),
            ReLU(name='tail_relu_2'),
            Dense(hidden_dim),
            Dropout(dropout_rate),
            ReLU(name='tail_relu_3'),
            Dense(1),
            Activation('tanh')
        ])

    def call(self, input, training=False):
        """
        
        Params:
            input: LIST of [positions, shape_idx] where positions is Bx3 and shape_idx is a scalar
        Returns:
            predicted sdf: [B,]
        """
        x, shape_idx = input
        # x, shape_idx = input
        # print("x in:", x[:6])
        shape_code = self.latent_shape_code_emb(shape_idx) #[shape_code_dim,]
        # print("shape code: ", shape_code.numpy()[:10])
        shape_code = tf.repeat(tf.expand_dims(shape_code, axis=0), tf.shape(x)[0], axis=0) # [B, shape_code_dim]
        # print("shape code shape:", shape_code.shape)
        input = tf.concat([shape_code, x], axis=1) # [B, (shape_code_dim+3)]
        intermediate = self.head(input) # [B, hidden-(shape_code_dim+3)]
        # skip connection
        intermediate = tf.concat([intermediate, input], axis=1) # [B, hidden_dim]
        out = self.tail(intermediate)

        # print("model out:", out.numpy()[:10])
        return tf.squeeze(out)

    @tf.function
    def loss(self, sdf_pred, sdf_true, clamp_dist):
        sdf_pred = tf.expand_dims(sdf_pred, axis=1)
        sdf_true = tf.expand_dims(sdf_true, axis=1)
        # print("model out: ", sdf_pred.numpy()[:5])
        # print("actual: ", sdf_true[:5])
        # return keras.losses.MeanAbsoluteError()(tf.clip_by_value(sdf_true, -1*clamp_dist, clamp_dist), tf.clip_by_value(sdf_pred, -1*clamp_dist, clamp_dist))
        return keras.losses.MeanAbsoluteError()(tf.clip_by_value(sdf_true, -1*clamp_dist, clamp_dist), tf.clip_by_value(sdf_pred, -1*clamp_dist, clamp_dist))

