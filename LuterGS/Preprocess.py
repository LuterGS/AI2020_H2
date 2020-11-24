from tqdm import tqdm
import numpy as np
import os
from konlpy.tag import Okt

# 파일 INIT
PATH = os.path.dirname(os.path.abspath(__file__))
TAG = Okt()
TAG_NAME = [
    "Noun",
    "Verb",
    "Adjective",
    "Determiner",
    "Adverb",
    "Conjunction",
    "Exclamation",
    "Josa",
    "PreEomi",
    "Eomi",
    "Suffix",
    "Punctuation",
    "Foreign",
    "Alpha",
    "Number",
    "Unknown",
    "KoreanParticle",
    "Hashtag",
    "ScreenName",
    "Email",
    "URL",
    "Modifier",
    "VerbPrefix"
    ]


# 파라미터로 입력받은 파일에 저장된 단어 리스트를 딕셔너리 형태로 저장
def load_vocab(filename):
    vocab_file = open(filename,'r',encoding='utf8')
    print("{} vocab file loading...".format(filename))

    # default 요소가 저장된 딕셔너리 생성
    symbol2idx, idx2symbol = {"<PAD>":0, "<UNK>":1}, {0:"<PAD>", 1:"<UNK>"}

    # 시작 인덱스 번호 저장
    index = len(symbol2idx)
    for line in tqdm(vocab_file.readlines()):
        symbol = line.strip()
        symbol2idx[symbol] = index
        idx2symbol[index]= symbol
        index+=1

    return symbol2idx, idx2symbol

# 입력 데이터를 고정 길이의 벡터로 표현하기 위한 함수
def convert_data2feature(data, symbol2idx, max_length=None):
    # 고정 길이의 0 벡터 생성
    feature = np.zeros(shape=(max_length), dtype=np.int)
    # 입력 문장을 공백 기준으로 split
    words = data.split()

    for idx, word in enumerate(words[:max_length]):
        if word in symbol2idx.keys():
            feature[idx] = symbol2idx[word]
        else:
            feature[idx] = symbol2idx["<UNK>"]
    return feature

# 파라미터로 입력받은 파일로부터 tensor객체 생성
def load_data(filename, word2idx, tag2idx):
    file = open(filename,'r',encoding='utf8')

    # return할 문장/라벨 리스트 생성
    indexing_inputs, indexing_tags = [], []

    print("{} file loading...".format(filename))

    # 실제 데이터는 아래와 같은 형태를 가짐
    # 문장 \t 태그
    # 세 종 대 왕 은 <SP> 조 선 의 <SP> 4 대 <SP> 왕 이 야 \t B_PS I_PS I_PS I_PS O <SP> B_LC I_LC O <SP> O O <SP> O O O
    for line in tqdm(file.readlines()):
        try:
            id, sentence, tags = line.strip().split('\t')
        except:
            id, sentence = line.strip().split('\t')
        input_sentence = convert_data2feature(sentence, word2idx, config["max_length"])
        indexing_tag = convert_data2feature(tags, tag2idx, config["max_length"])

        indexing_inputs.append(input_sentence)
        indexing_tags.append(indexing_tag)

    return np.array(indexing_inputs), np.array(indexing_tags)

# tensor 객체를 리스트 형으로 바꾸기 위한 함수
def tensor2list(input_tensor):
    return input_tensor.cpu().detach().numpy().tolist()


def convert_data2tagfeature(sentence, max_length, embedding_dim):
    """
    문장을 품사 정보를 토대로 벡터링
    예를 들어, "나는 건국대 학생" 이라는 단어가 있으면, 이걸 pos로 매핑하면
    (나, noun), (는, Josa), (건국대, Noun), (학생, Noun) 으로 바꿔주는데,
    n
    :param sentence:
    :param max_length:
    :param embedding_dim:
    :return:
    """

    result = [np.full(shape=embedding_dim, fill_value=0, dtype=np.float) for i in range(max_length)]
    pos = TAG.pos(sentence)
    i = 0

    for morphs in pos:
        num = TAG_NAME.index(morphs[1]) + 1
        # print(morphs, num, i)
        for letter in morphs[0]:
            result[i] = np.full(shape=embedding_dim, fill_value=num, dtype=np.float)
            i += 1
            # print("normal, ", i)
            if i == max_length:
                break

        if i == max_length:
            break

        try:
            if sentence[i] == " ":
                result[i] = np.full(shape=embedding_dim, fill_value=0, dtype=np.float)
                i += 1
                # print('TRY : ', i)
                if i == max_length:
                    break
        except IndexError:
            break

    if len(result) != max_length:
        print("데이터 전처리가 잘못되었습니다! 프로그램을 종료합니다 :", sentence)
        exit(1)

    return np.asarray(result)


def get_one_posdata(filepath, embedding=14, max_length=150):
    """
    :param filepath: 전처리 데이터를 얻을 파일 경로입니다. (상대경로시 이 함수를 호출하는 파일의 경로를 시작점으로 합니다)
    :param embedding: biGRU 계층을 지난 결과값고 concat하기 때문에, 계층의 데이터를 균등하기 주기 위해 차원을 임의로 늘렸습니다.
                        늘릴 차원의 개수입니다.
    :param max_length: 문장의 최대 길이입니다. 기본값 150으로 설정되어 있으며, 값을 변경할 수 있습니다.

    예시)
    먼저, max_length가 20이라고 가정하면, 크기가 20인 빈 numpy array를 만듭니다.
    이후, 나는 건국대 학생이다.  라는 12글자의 문장이 있다고 할 때, 해당 글자를 Okt를 이용해 pos를 추출합니다.
    (나, noun), (는, josa), (건국대, noun), (학생, noun), (이다, josa), (. ,Punctuation) 으로 분리가 됩니다. (예시이며, 다르게 나올 수 있음)
    noun을 1, josa를 2, punctuation을 3, space를 0이라고 하면, 해당 문장을 다음과 같은 벡터로 치환합니다.
    나   는   <SP>    건   국   대   <SP>    학   생   이   다   .
    1    2   0        1    1    1   0       1   1    2   2    3     0...
    -> [12011101122300000000]       -> 벡터 1
    이후, 차원을 늘려주기 위해, 각 값과 같은 크기를 가지는 embedding (기본 14)만큼의 numpy array를 채워넣습니다.
    즉, 위 벡터에서 1은 [1 1 1 1 1 1 1 1 1 1 1 1 1 1] 로 변환됩니다.
    따라서, 위 벡터는 이런 값으로 변환됩니다.
    [[1111....
     [2222....
     [0000...
     [1111...
     [1111...
     [1111...
     [0000...
     [1111...
     [1111...
     [2222...
     [2222...
     [3333...
     [0000...
     ...]       -> 벡터 2

    :return: [문장 개수][max_length][embedding] 의 크기를 가지는 numpy array를 return 합니다.
    """
    result = []
    with open(filepath, "r", encoding="utf8") as file:
        for line in tqdm(file.readlines()):
            line = line.split("\t")

            if len(line) == 3:
                raw_sentence = line[1]
            else:
                raw_sentence = line[0]

            result.append(convert_data2tagfeature(raw_sentence.replace(" ", "").replace("<SP>", " "), max_length=max_length, embedding_dim=embedding))

    return np.asarray(result)



if __name__ == "__main__":

    # np.save("ner_dev", get_one_posdata("../baseline/ner_dev.txt"))
    # np.save("ner_train", get_one_posdata("../baseline/ner_train.txt"))

    dev = np.load("ner_dev.npy")
    print(dev.shape)
    train = np.load("ner_train.npy")
    print(train.shape)


