#%% [markdown]
# # SoyBean Predictions

#%%
import pandas as pd
import numpy as np
import sklearn as sl
from collections import defaultdict

#%% [markdown]
# ## Load and Transform Data
#%% [markdown]
# ### Create Column Names

#%%
Columns = ['Disease','date','plant-stand','precip','temp','hail','crop-hist','area-damaged','severity','seed-tmt',
           'germination','plant-growth','leaves','leafspots-halo','leafspots-marg','leafspots-size','leaf-shread',
           'leaf-malf','leaf-mild','stem','lodging','stem-cankers','canker-lesion','fruiting-bodies','external decay',
           'mycelium','int-discolor','sclerotia','fruit-pods','fruit spots','seed','mold-growth','seed-discolor',
           'seed-size','shriveling','roots']

#%% [markdown]
# one thing to note, the missing values in this dataset are denoted by a "?", so we will be changing that to a NaN type
#%% [markdown]
# ### Read Data

#%%
soybeansU = pd.read_csv('Soybean.csv')
soybeansU.columns = Columns
soybeansU = soybeansU.replace('?',np.nan)

#%% [markdown]
# Now lets check the info about this data

#%%
soybeansU.info()
soybeansU.shape

#%% [markdown]
# So the dataframe is 306 rows by 36 columns

#%%
soybeansU.head(35) # Check to make sure it was imported prpperly 

#%% [markdown]
# If you look at the data you notice all the of data are numerical. This is actually categories decoded by numbers. 0 to represent the first category, 1 to represent the 2 category and so on. This is the breakdown
# 
#    1. date:april,may,june,july,august,september,october,?.
#    2. plant-stand:	normal,lt-normal,?.
#    3. precip:		lt-norm,norm,gt-norm,?.
#    4. temp:		lt-norm,norm,gt-norm,?.
#    5. hail:		yes,no,?.
#    6. crop-hist:	diff-lst-year,same-lst-yr,same-lst-two-yrs,same-lst-sev-yrs,?.
#    7. area-damaged:	scattered,low-areas,upper-areas,whole-field,?.
#    8. severity:	minor,pot-severe,severe,?.
#    9. seed-tmt:	none,fungicide,other,?.
#    10. germination:	90-100%,80-89%,lt-80%,?.
#    11. plant-growth:	norm,abnorm,?.
#    12. leaves:		norm,abnorm.
#    13. leafspots-halo:	absent,yellow-halos,no-yellow-halos,?.
#    14. leafspots-marg:	w-s-marg,no-w-s-marg,dna,?.
#    15. leafspot-size:	lt-1/8,gt-1/8,dna,?.
#    16. leaf-shread:	absent,present,?.
#    17. leaf-malf:	absent,present,?.
#    18. leaf-mild:	absent,upper-surf,lower-surf,?.
#    19. stem:		norm,abnorm,?.
#    20. lodging:    	yes,no,?.
#    21. stem-cankers:	absent,below-soil,above-soil,above-sec-nde,?.
#    22. canker-lesion:	dna,brown,dk-brown-blk,tan,?.
#    23. fruiting-bodies:	absent,present,?.
#    24. external decay:	absent,firm-and-dry,watery,?.
#    25. mycelium:	absent,present,?.
#    26. int-discolor:	none,brown,black,?.
#    27. sclerotia:	absent,present,?.
#    28. fruit-pods:	norm,diseased,few-present,dna,?.
#    29. fruit spots:	absent,colored,brown-w/blk-specks,distort,dna,?.
#    30. seed:		norm,abnorm,?.
#    31. mold-growth:	absent,present,?.
#    32. seed-discolor:	absent,present,?.
#    33. seed-size:	norm,lt-norm,?.
#    34. shriveling:	absent,present,?.
#    35. roots:		norm,rotted,galls-cysts,?.
# 
#%% [markdown]
# ### Handling missing values
#%% [markdown]
# Lets check is there is missing values

#%%
soybeansU.isna().sum()

#%% [markdown]
# There seems to be quite a bit of missing values, with the coloumn with the highest amount of missing values being lodging
#%% [markdown]
# There are 19 classes and they are described as such:
# 
# Class Distribution: 
# 1. diaporthe-stem-canker: 10
# 2. charcoal-rot: 10
# 3. rhizoctonia-root-rot: 10
# 4. phytophthora-rot: 40
# 5. brown-stem-rot: 20
# 6. powdery-mildew: 10
# 7. downy-mildew: 10
# 8. brown-spot: 40
# 9. bacterial-blight: 10
# 10. bacterial-pustule: 10
# 11. purple-seed-stain: 10
# 12. anthracnose: 20
# 13. phyllosticta-leaf-spot: 10
# 14. alternarialeaf-spot: 40
# 15. frog-eye-leaf-spot: 40
# 16. diaporthe-pod-&-stem-blight: 6
# 17. cyst-nematode: 6
# 18. 2-4-d-injury: 1
# 19. herbicide-injury: 4
#%% [markdown]
# Let's map these classes. 

#%%
class_mapping = {label:idx for idx, label in 
                 enumerate(np.unique(soybeansU["Disease"]), start = 1)}
class_mapping

#%% [markdown]
# As you see the right side of each class is their new mapping starting from 1 and ending at 19
# 
# Now let's apply it to the column

#%%
soybeansU['Disease'] = soybeansU['Disease'].map(class_mapping)
soybeansU.head(10)

#%% [markdown]
# As you can see, the column `Disease` is now numbers.
# now we will impute the NaNs with the most frequen class in each coloumn

#%%
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan, strategy= 'most_frequent')
imp = imp.fit(soybeansU)
imputed_data = imp.transform(soybeansU.values)
imputed_data


