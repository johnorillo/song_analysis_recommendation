import re
import nltk

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


def playlist_synonyms(text):
    new_list = tokenize_synonyms_hypernyms(text,6)
    return ' '.join(new_list)


def tokenize_synonyms_hypernyms(text, max_words=5):
    # tokenize
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    extended_tokens = []
    synonyms_tokens = []
    hypernyms_tokens = []
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    # remove stop words
    stopwords = nltk.corpus.stopwords.words('english')
    filtered_tokens_no_stop = [x for x in filtered_tokens if x not in stopwords]

    # check synonyms and hypernyms per filtered tokens
    if len(filtered_tokens_no_stop) != 0:
        for filter_token in filtered_tokens_no_stop:
            synonyms_tokens.extend(get_synonyms(filter_token,max_words))
            synonyms_tokens.extend(get_hypernyms(filter_token,max_words))
        filtered_tokens_no_stop.extend(synonyms_tokens)
        extended_tokens = filtered_tokens_no_stop
    else:
        extended_tokens = filtered_tokens_no_stop


    return list(set(extended_tokens))


def get_synonyms(token, max_out):
    synonyms = []
    sys = wordnet.synsets(token)
    for sys_element in sys:
        for l in sys_element.lemmas():
            synonyms.append(l.name())
    return synonyms[:max_out]


def get_hypernyms(token, max_out):
    hypernyms = []
    sys = wordnet.synsets(token)
    for sys_element in sys:
        for h in sys_element.hypernyms():
            hypernyms.append(h.name().split('.')[0])
    return hypernyms[:max_out]


def sampler_df(df_orig, rand_state,split_perc):
    df_orig.reset_index(drop=True,inplace=True)
    split_1 = int(len(df_orig)*split_perc)
    df_test = df_orig.sample(n=split_1,replace=False,random_state=rand_state)
    df_train = df_orig.drop(index=list(df_test.index))

    df_test.reset_index(drop=True,inplace=True)
    df_train.reset_index(drop=True,inplace=True)
    return df_train,df_test