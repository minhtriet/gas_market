import subprocess

def create_input():
    # read all news paper, extract abstract or info
    # return triplet with properties
    pass


def process_entity_relations(entity_relations_str):
    # format is ollie.
    entity_relations = list()
    for s in entity_relations_str:
        entity_relations.append(s[s.find("(") + 1:s.find(")")].split(';'))
    return entity_relations

create_input()
subprocess.call(['java', '-jar', 'Blender.jar'])
# classify_envents