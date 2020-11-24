import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras import Model
from tf2crf import CRF, ModelWithCRFLoss

def biGRU_ResNet_Okt(MAX_LENGTH, EMBEDDING_DIM, TAG_NUM):

    feature_input = layers.Input(shape=(MAX_LENGTH, ), dtype='int32')
    f_x = layers.Embedding(input_dim=2160, output_dim=EMBEDDING_DIM, input_shape=(MAX_LENGTH, ), trainable=True, mask_zero=True)(feature_input)
    f_x = layers.Bidirectional(layers.GRU(100, return_sequences=True))(f_x)
    f2_x = layers.Bidirectional(layers.GRU(units=100, return_sequences=True))(f_x)

    f_x = layers.Concatenate()([f_x, f2_x])
    f_x = layers.Dropout(0.2)(f_x)
    f_x = layers.Dense(TAG_NUM, activation='relu')(f_x)

    tag_input = layers.Input(shape=(MAX_LENGTH, TAG_NUM), dtype='float32')
    t_x = layers.Dense(units=TAG_NUM, activation='relu')(tag_input)

    x = layers.Concatenate()([f_x, t_x])
    x = layers.Dense(units=14, activation='relu')(x)

    crf = CRF(dtype='float32')
    x = crf(x)

    base_model = Model(inputs=[feature_input, tag_input], outputs=x)
    utils.plot_model(base_model, "biGRU+ResNet+Okt.png", True, True, 'TB', True, 120)

    model = ModelWithCRFLoss(base_model)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005))
    return model

    # f1-score : micro agv 0.68, macro avg 0.7, weighted avg 0.68
    # Adam, lr 0.0005, epoch 30, batch size 20

    # f1-score : micro agv 0.64, macro avg 0.59, weighted avg 0.64
    # Adam, lr 0.0005, epoch 100, batch size 20



def network2(MAX_LENGTH, EMBEDDING_DIM, TAG_NUM):

    feature_input = layers.Input(shape=(MAX_LENGTH, ), dtype='int32')
    f_x = layers.Embedding(input_dim=2160, output_dim=EMBEDDING_DIM, input_shape=(MAX_LENGTH, ), trainable=True, mask_zero=True)(feature_input)
    f_x = layers.Bidirectional(layers.GRU(units=100, return_sequences=True))(f_x)
    f2_x = layers.Bidirectional(layers.GRU(units=100, return_sequences=True))(f_x)
    f3_x = layers.Bidirectional(layers.GRU(units=100, return_sequences=True))(f2_x)

    f_x = layers.Concatenate()([f_x, f2_x, f3_x])
    f_x = layers.Dropout(0.2)(f_x)
    f_x = layers.Dense(TAG_NUM, activation='relu')(f_x)

    tag_input = layers.Input(shape=(MAX_LENGTH, TAG_NUM), dtype='float32')
    t_x = layers.Dense(units=TAG_NUM, activation='relu')(tag_input)

    x = layers.Concatenate()([f_x, t_x])
    x = layers.Dense(units=14, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    crf = CRF(dtype='float32')
    x = crf(x)

    base_model = Model(inputs=[feature_input, tag_input], outputs=x)
    base_model.summary()
    utils.plot_model(base_model, "biGRU+3Resnet+Okt.png", True, True, 'TB', True, 120)

    model = ModelWithCRFLoss(base_model)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005))
    return model

    # f1-score : micro agv 0.56, macro avg 0.45, weighted avg 0.55
    # Adam, lr 0.0005, epoch 20, batch size 20

    # f1-score : micro agv 0.67, macro avg 0.55, weighted avg 0.66
    # Adam, lr 0.001, epoch 20, batch size 20


