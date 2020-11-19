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

    result = [np.full(shape=embedding_dim, fill_value=0, dtype=np.float) for i in range(150)]
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

    if len(result) != 150:
        print(sentence)
        exit(1)

    return np.asarray(result)




