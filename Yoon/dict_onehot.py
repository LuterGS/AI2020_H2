def load_dict(f_name):
    print("{} dict file loading...".format(f_name))
    with open(os.path.join(root_dir + 'data', f_name),'r',encoding='utf8') as f:
        dictionary = []
        for line in f.readlines():
            line = re.sub("\n", "", line)
            line = re.sub("<SP>", "", line)
            line = re.sub("(주)", "", line)
            line = line.replace(" ", "")
            dictionary.append(line)
    
    return set(dictionary)


def convert_sent2dict(sent, max_length):  # baseline load_data 함수에서의 sentence 변수가 첫번째 인자로 그대로 들어가면 됨
    sent = sent.replace(" ","")
    sent = sent.replace("<SP>"," ")  # 입력 문장을 일반적인 문장 형태로 변환

    original_length = len(sent)  

    dict_onehot = []
    words = sent.split()  # 문장을 띄어쓰기 기준으로 분리

    for word in words:
        loc_tag, org_tag, per_tag = 0, 0, 0

        lookup = ""  # 사전에서 검색을 할 음절 조합
        match_count = 0
        for i, char in enumerate(word):
            lookup += char  # 단어에서 음절을 하나씩 늘려감
            if lookup in per_dict:  # PER.txt를 load_dict에 넣어서 나온 값을 담은 전역변수
                loc_tag, org_tag, per_tag = 0, 0, 0  # 기존 원핫을 초기화
                per_tag = 1  
                match_count = i + 1
            elif lookup in loc_dict:
                loc_tag, org_tag, per_tag = 0, 0, 0  
                loc_tag = 1
                match_count = i + 1
            elif lookup in org_dict:
                loc_tag, org_tag, per_tag = 0, 0, 0
                org_tag = 1
                match_count = i + 1
        if match_count > 1:  # 사전에서 매치된 음절이 2개 이상일 경우
            for _ in range(match_count):  # 매치된 음절까지만 원핫을 append
                dict_onehot.append([loc_tag, org_tag, per_tag])

            for _ in range(len(word) - match_count):  # 해당 단어에서 매치되지 않은 나머지 음절들 (조사 등..) 
                dict_onehot.append([0, 0, 0])  
        else:  # 사전에서 매치된 음절의 길이가 1 이하일 경우
            for _ in range(len(word)):
                dict_onehot.append([0, 0, 0])
        
        dict_onehot.append([0, 0, 0])  # 단어 간 띄어쓰기

    while len(dict_onehot) < max_length:  # 패딩
        dict_onehot.append([0, 0, 0])

    return dict_onehot[:config['max_length']]
