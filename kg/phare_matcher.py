import spacy
import owlready2

owlready2.onto_path.append("/path/to/your/local/ontology/repository")
onto.load()

def entity_recognition(s):
    doc = nlp(s)
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)


nlp = spacy.load('en_core_web_lg')
s = "Russia's Gazprom warns Europe it could face gas shortages"
entity_recognition(s)
