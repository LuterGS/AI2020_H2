# 아이디어
# 1. 기존 baseline에서 했던 대로 120차원으로 인코딩
# 2. Okt를 사용해 각 문장에서의 품사를 매핑
# 두 개를 mixing


import os
from tqdm import tqdm
import numpy as np
import LuterGS.model as models
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

    def __init__(self, model_selection):
        self.model = self.init_model(model_selection)

    def init_model(self, model_selection):
        """
        :param model_selection: 모델 선택 번호
            1. biGRU + Resnet + Okt 모델
        :return: 완성된 텐서플로우 모델
        """
        if model_selection == 1:
            return models.biGRU_ResNet_Okt(MAX_LENGTH, EMBEDDING_DIM, TAG_NUM)
        elif model_selection == 2:
            return models.network2(MAX_LENGTH, EMBEDDING_DIM, TAG_NUM)

    def train(self, x1, x2, y, save_name, epochs=30, batch_size=20):
        self.model.fit(x=[x1, x2], y=y, epochs=epochs, batch_size=batch_size, verbose=1)
        self.model.save(filepath=save_name)

    def load(self, location):
        self.model.load_weights(location + "/variables")

    def evaluate(self, x1, x2, y, idx2tag):
        model_predicted = self.model.predict([x1, x2])

        answers, predicts = [], []
        for idx, answer in enumerate(y):
            answers.extend([idx2tag[e].replace("_", "-") for e in answer if idx2tag[e] != "<SP>" and idx2tag[e] != "<PAD>"])
            predicts.extend([idx2tag[e].replace("_", "-") for i, e in enumerate(model_predicted[0][idx]) if idx2tag[answer[i]] != "<SP>" and idx2tag[answer[i]] != "<PAD>"])

        print(classification_report(answers, predicts))

if __name__ == "__main__":

    neural_net = Network(2)


    data = Dataset()
    train_x1, train_x2, train_y = data.get_network_data("../baseline/ner_train.txt")
    test_x1, test_x2, test_y = data.get_network_data("../baseline/ner_dev.txt")

    neural_net.train(train_x1, train_x2, train_y, "network2", 30, 20)
    # model1은 biGRU_ResNet_Okt에서 두개의 BiGRU를 합칠때 Add 사용
    # model2는 Concatenate 사용
    # network2는 models.network2 사용

    neural_net.evaluate(test_x1, test_x2, test_y, data.idx2tag)


