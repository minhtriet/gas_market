"""
A zoo of word embedding
"""
import spacy
# nlp = spacy.load('en_core_web_lg', disable=['ner'])
nlp = spacy.load('en_core_web_lg')
for word in nlp.Defaults.stop_words:
    for w in (word, word[0].upper() + word[1:], word.upper()):
        lex = nlp.vocab[w]
        lex.is_stop = True
print('Load FastText')
# import sent2vec
# model = sent2vec.Sent2vecModel()
# model.load_model(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'twitter_bigrams.bin'))
# model = 0
print('FastText completed')


def fasttext(s):
    return 0


#    return model.embed_sentence(s).flatten()


def filter_tweet(tweets):
    filtered_tweet = []
    for x in tweets:
        if 'quoted_status_id' in x:
            continue
        if 'retweeted_status' in x:
            filtered_tweet.append(x['retweeted_status'])
        else:
            if x['in_reply_to_status_id'] == 'null':
                print('original %s %s' % (x['id'], x['text']))
                filtered_tweet.append(x)
    return filtered_tweet


def spacy(s):
    doc = nlp(s)
    # stopwords, punctuation and number
    necessary_word = []
    for word in doc:
        if not (word.is_stop or word.is_punct):
            necessary_word.append(word)
    doc = [word.lemma_ for word in necessary_word]
    return nlp(' '.join(doc)).vector


def filter_tweet(tweets):
    filtered_tweet = []
    for x in tweets:
        if 'quoted_status_id' in x:
            continue
        if 'retweeted_status' in x:
            filtered_tweet.append(x['retweeted_status'])
        else:
            if x['in_reply_to_status_id'] == 'null':
                print('original %s %s' % (x['id'], x['text']))
                filtered_tweet.append(x)
    return filtered_tweet
