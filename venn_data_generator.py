import pandas as pd
from os import path
"""
use reverb in sentence
count sentence without verbs
engage python pipeline
"""

"""
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
 
# Make the diagram
venn3(subsets = (a_only, b_only, a_and_b, c_only, a_and_c, b_and_c, abc))
plt.show()
"""

# read all
nytimes = pd.read_csv(path.join('nytimes', 'data_exp.csv'), sep=';', encoding='utf-8', names=['pub_date', 'info'])
guardian = pd.read_csv(path.join('guardian', 'data.csv'), encoding='utf-8', names=['pub_date', 'info'])
ft = pd.read_csv(path.join('ft', 'data.csv'), encoding='utf-8', names=['pub_date', 'info'])

with open('all_headlines.txt', 'w', encoding='utf-8') as f:
    for x in [nytimes, guardian, ft]:
        for sentence in x['info']:
            f.write(f'{sentence}\n')
            
