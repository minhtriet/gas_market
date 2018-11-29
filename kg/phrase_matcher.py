import spacy
import owlready2 as owl


def entity_recognition(s):
    doc = nlp(s)
    nes = []
    pos = []
    # pos
    for token in doc:
        print(token.pos_)
    # ner
    for ent in doc.ents:
        pos.append(ent.pos_)
        nes.append(ent.label_)
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
    return nes, pos


nlp = spacy.load('en_core_web_lg')
onto = owl.get_ontology('cevo.owl').load()
entities = list(onto.classes())
individuals = list(onto.individuals())

ss = ["Russia's Gazprom warns Europe it could face gas shortages"]
for s in ss:
    nes, pos = entity_recognition(s)

    verb = [x.name for x in individuals]
    for p in pos:
        if pos in individuals:
            # map to class
            # owl.entity.ThingClass
            pass