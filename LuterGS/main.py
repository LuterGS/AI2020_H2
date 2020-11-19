# 아이디어
# 1. 기존 baseline에서 했던 대로 120차원으로 인코딩
# 2. Okt를 사용해 각 문장에서의 품사를 매핑
# 두 개를 mixing


import os
from tqdm import tqdm
import numpy as np

import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import tensorflow.keras.metrics as metrics
import tensorflow.keras.utils as utils
from tensorflow.keras import Model
from tf2crf import CRF, ModelWithCRFLoss
from seqeval.metrics import classification_report


from LuterGS.Preprocess import load_vocab, convert_data2feature, convert_data2tagfeature

PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
MAX_LENGTH = 150
EMBEDDING_DIM = 100
TAG_NUM = 14

class Dataset:

    def __init__(self):
        self.word2idx, self.idx2word = load_vocab("../baseline/vocab.txt")
        self.tag2idx, self.idx2tag = load_vocab("../baseline/tag_vocab.txt")


    def get_network_data(self, filename):
        feature, pos, y = [], [], []
        with open(PATH + filename, "r", encoding="utf8") as file:
            for line in tqdm(file.readlines()):
                line = line.split("\t")

                if len(line) == 3:
                    raw_id = line[0]
                    raw_sentence = line[1]
                    raw_label = line[2]

                else:
                    raw_sentence = line[0]
                    raw_label = line[1]

                feature.append(convert_data2feature(raw_sentence, self.word2idx, MAX_LENGTH))
                y.append(convert_data2feature(raw_label, self.tag2idx, MAX_LENGTH))
                pos.append(convert_data2tagfeature(raw_sentence.replace(" ", "").replace("<SP>", " "), MAX_LENGTH, TAG_NUM))

        return np.asarray(feature), np.asarray(pos), np.asarray(y)


class Network:

    def __init__(self):
        self.model = self.init_model()

    def init_model(self):
        feature_input = layers.Input(shape=(MAX_LENGTH, ), dtype='int32')
        f_x = layers.Embedding(input_dim=2160, output_dim=EMBEDDING_DIM, input_shape=(MAX_LENGTH, ), trainable=True, mask_zero=True)(feature_input)
        f_x = layers.Bidirectional(layers.GRU(100, return_sequences=True))(f_x)

        f2_x = layers.Bidirectional(layers.GRU(units=100, return_sequences=True))(f_x)

        f_x = layers.Add()([f_x, f2_x])

        f_x = layers.Dropout(0.2)(f_x)
        f_x = layers.Dense(TAG_NUM, activation='relu')(f_x)

        tag_input = layers.Input(shape=(MAX_LENGTH, TAG_NUM), dtype='float32')
        t_x = layers.Dense(units=TAG_NUM, activation='relu')(tag_input)

        x = layers.Concatenate()([f_x, t_x])
        x = layers.Dense(units=14, activation='relu')(x)
        # x = layers.Reshape(target_shape=(1, 150, TAG_NUM * 2))(x)
        # x = layers.Conv2D(filters=64, kernel_size=4, strides=1, padding='SAME', activation='relu', kernel_initializer='he_normal')(x)
        # x = layers.Reshape(target_shape=(150, TAG_NUM))(x)

        crf = CRF(dtype='float32')
        x = crf(x)


        base_model = Model(inputs=[feature_input, tag_input], outputs=x)
        utils.plot_model(base_model, "biGRU+ResNet+Okt.png", True, True, 'TB', True, 120)

        model = ModelWithCRFLoss(base_model)
        model.compile(optimizer=optimizers.Adam(lr=0.0003))
        return model

    def train(self, x1, x2, y, save_name, epochs=20, batch_size=32):
        self.model.fit(x=[x1, x2], y=y, epochs=epochs, batch_size=batch_size, verbose=1)
        self.model.save(filepath=save_name)

    def evaluate(self, x1, x2, y, idx2tag):
        model_predicted = self.model.predict([x1, x2])

        answers, predicts = [], []
        for idx, answer in enumerate(y):
            answers.extend([idx2tag[e].replace("_", "-") for e in answer if idx2tag[e] != "<SP>" and idx2tag[e] != "<PAD>"])
            predicts.extend([idx2tag[e].replace("_", "-") for i, e in enumerate(model_predicted[0][idx]) if idx2tag[answer[i]] != "<SP>" and idx2tag[answer[i]] != "<PAD>"])

        print(classification_report(answers, predicts))

if __name__ == "__main__":

    neural_net = Network()


    data = Dataset()
    train_x1, train_x2, train_y = data.get_network_data("../baseline/ner_train.txt")
    test_x1, test_x2, test_y = data.get_network_data("../baseline/ner_dev.txt")

    neural_net.train(train_x1, train_x2, train_y, "model2", 20)
    neural_net.evaluate(test_x1, test_x2, test_y, data.idx2tag)


