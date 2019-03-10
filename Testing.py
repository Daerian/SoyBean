import pandas as pd 
import numpy as np
from collections import defaultdict

Columns = ['Disease','date','plant-stand','precip','temp','hail','crop-hist','area-damaged','severity','seed-tmt',
           'germination','plant-growth','leaves','leafspots-halo','leafspots-marg','leafspots-size','leaf-shread',
           'leaf-malf','leaf-mild','stem','lodging','stem-cankers','canker-lesion','fruiting-bodies','external decay',
           'mycelium','int-discolor','sclerotia','fruit-pods','fruit spots','seed','mold-growth','seed-discolor',
           'seed-size','shriveling','roots']

        
soybeansU = pd.read_csv('Soybean.csv')
soybeansU.columns = Columns
soybeansU = soybeansU.replace('?',np.nan)


soybeansU.head(35) # Check to make sure it was imported prpperly 
soybeansU = soybeansU.fillna(method = 'pad')
soybeansU.head(35) # Check to make sure it was imported prpperly 


diseases = soybeansU.Disease.unique()
soybeans = pd.DataFrame(columns=Columns)

for disease in diseases:
    subset = soybeansU[soybeansU['Disease'].str.contains(disease)]
    subset = subset.fillna(999999)
    print(subset.head(10))

    soybeans = soybeans.append(subset, ignore_index=True)
        

soybeans.head(35)
