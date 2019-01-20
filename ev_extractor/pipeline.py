import os
import re
from contextlib import contextmanager
from copy import deepcopy
from os import path
from subprocess import call

import spacy
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError

nlp = spacy.load('en_core_web_lg')


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def _extract(sent):
    doc = nlp(sent)
    print('Extracting clauses and phrases')
    # subordinate phrases
    splits = []
    clauses = []
    for stn in doc.sents:
        for word in stn:
            if len(list(word.subtree)) > 1:  # Cuadrilla pauses fracking operations after tremor in Lancashire site
                if word.dep_ in ('xcomp', 'ccomp', 'rcmod', 'advcl') or word.pos_ in ('ADP'):  # phrase
                    print('Phrase: ', word, list(word.subtree))
                    phrase = ' '.join(w.text_with_ws.strip() for w in word.subtree)
                    phrase = re.sub('\s(?=[,:])', '', phrase)
                    splits.append(phrase)
                if word.pos_ in ('VERB'):  # clause VERB
                    print('Clause: ', word, list(word.subtree))
                    clause = ' '.join(w.text_with_ws.strip() for w in word.subtree)
                    clause = re.sub('\s(?=[,:])', '', clause)
                    clauses.append(clause)
    print('Clauses: ', clauses)
    print('Phrases: ', splits)
    print('Extraction done')
    return clauses, splits


def _consolidate(splits):
    new_split = deepcopy(splits)
    for i in range(len(splits)):
        for j in range(len(splits)):
            if i != j:
                new_split[i] = new_split[i].replace(splits[j], '').strip()
    print('Consolidate splits: ', new_split)
    return new_split


def _lemmatize(doc):
    return [token.lemma_ for token in doc if not token.is_stop]


def _informative_filter(split):
    """
    delete clauses, delete the word natural before the word gas, count
    adj and noun
    :param splits:
    :return:
    """
    print(split)
    if len(split) > 0:
        temp_split = split
        temp_split = temp_split.replace('natural gas', 'gas')
        doc = nlp(temp_split)
        for ent in doc.ents:
            if ent.label_ == 'GPE':
                print(True)
                return True
        for token in doc:
            if token.pos_ == 'ADJ':
                print(True)
                return True
    print(False)
    return False


def pipeline(sent):
    print(sent)
    clauses, phrases = _extract(sent)
    if len(phrases) > 0:
        with cd(path.join('lib', 'ims_0.9.2.1')):
            with open('test_.txt', 'w', encoding='utf-8') as f:
                # if clauses, ie has verb, no need to go through pipeline
                for split in phrases:
                    f.write('%s\n' % split)
            call(['bash', 'testPlain.bash', 'models-MUN-SC-wn30', 'test_.txt', 'out_.txt',
                  path.join('lib', 'dict', 'index.sense')])
    phrases = _disambiguation()
    print('Check informative')
    phrases[:] = [tup for tup in phrases if _informative_filter(tup)]
    print('Checked informative')
    print(phrases)
    print('-----')
    print([clauses], [phrases])
    clauses.extend(phrases)
    print(clauses)
    print('Consolidate')
    clauses = _consolidate(clauses)
    print(clauses)
    print('Final ', clauses)
    return clauses


def _disambiguation():
    ev = ['noun.phenomenon', 'noun.act', 'noun.event', 'noun.attribute', 'adj.all', 'adv.all']
    true_splits = []
    with cd(path.join('lib', 'ims_0.9.2.1')):
        pattern = '<x(.+?)</x>'
        with open('out_.txt', 'r', encoding='utf-8') as f, open("test_.txt", 'r', encoding='utf-8') as f1:
            for line, line1 in zip(f, f1):
                matches = re.finditer(pattern, line)
                lexnames = []
                for m in matches:
                    key = re.search('(?<=\s)([^ ]+?)(?=\|)', m[0]).group(0)  # for '   natural%3:00:03::|'
                    try:
                        lexname = wn.lemma_from_key(key).synset().lexname()
                        lexnames.append(lexname)
                    except WordNetError:
                        print(key)
                print(lexnames)
                print(line1)
                if set(lexnames).intersection(set(ev)):
                    true_splits.append(line1.strip())
    print('Disambiguation: ', true_splits)
    return true_splits
