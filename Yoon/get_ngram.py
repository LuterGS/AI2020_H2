from itertools import product
from collections import OrderedDict

def get_ngram(word, min=2, max=15):
    ngrams = []

    word = word[:max]  # 단어 하나의 길이가 너무 길 경우, 조합의 수가 너무 많아짐

    for comb in product([0, 1], repeat=len(word)-1):
        split_index = [0]
        for i in range(len(comb)):
            if comb[i] == 1:
                split_index.append(i+1)
        
        split_index.append(None)
        ngram = [word[split_index[i]:split_index[i+1]] for i in range(len(split_index)-1)]
        
        for ng in ngram:
            if min <= len(ng) <= max:
                ngrams.append(ng)
    
    ngrams = list(OrderedDict.fromkeys(ngrams))
    ngrams.sort(key=len)
    return ngrams


# 입력: "홍길동이다"
# 출력: ['이다', '홍길', '동이', '길동', '홍길동', '동이다', '길동이', '홍길동이', '길동이다', '홍길동이다']
# 출력의 순서는 매번 같습니다. 같은 ngram이 매번 같은 인덱스에 위치함
# 리턴되는 ngram의 길이는 min이상 max 이하의 것들이고, 단어 자체가 max보다 길면 max에서  잘립니다

