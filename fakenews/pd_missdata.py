''' Some basic usage of Pandas to identify and visualize the data '''
import pandas as pd
import numpy as np


DATA = pd.DataFrame({'Animal': ['Pelican', 'Duck', 'Chicken'],
                     'Weight': [None, 3.5, 2.54],
                     'Height': [4.3, np.nan, 0.5],
                     'Habitat': ['??', '*', 'Backyard']})
print('Previewing the data:')
print(DATA)
print()

print('Where is missing:')
print(DATA.isna())
print()

print(f'Missing values:\n{DATA.isna().sum()}\n')

# replacing
DATA['Habitat'].replace('??', np.nan, inplace=True)
miss = {'Weight': np.nan, 'Height': np.nan, 'Habitat': ['??', '*']}
DATA.replace(np.nan, value=999, inplace=True)
print(f'Data with replacement: \n{DATA}')
print()
