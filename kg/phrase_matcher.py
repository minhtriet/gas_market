import spacy
import ontospy


def extract_relation(doc):
    # merge entities and noun chunks into one token
    # spans = list(doc.ents) + list(doc.noun_chunks)
    # for span in spans:
    #     span.merge()

    relations = []
    for money in doc:
        if money.dep_ in ('attr', 'dobj'):
            subject = [w for w in money.head.lefts if w.dep_ == 'nsubj']
            if subject:
                subject = subject[0]
                relations.append((subject, money))
        elif money.dep_ == 'pobj' and money.head.dep_ == 'prep':
            relations.append((money.head.head, money))

    return relations


def entity_recognition(s):
    nes = []
    # ner
    for ent in s.ents:
        nes.append(ent.label_)
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
    return nes


nlp = spacy.load('en_core_web_lg')
model = ontospy.Ontospy('cevo.owl', verbose=True)
# entities = list(onto.classes())
# individuals = list(onto.individuals())

ss = ["Russia's Gazprom warns Europe it could face gas shortages",
      "In addition, its machines are typically easier to operate, so customers require less assistance from software",
      "The market for system-management software for Digital's hardware is fragmented enough that a giant such as Computer Associates should do well there"]
for s in ss:
    print(s)
    doc = nlp(s)
    nes = entity_recognition(doc)
    print(extract_relation(doc))
    # verb = [x.name for x in individuals]
    pass
