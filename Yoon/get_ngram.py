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
