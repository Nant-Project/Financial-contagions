import numpy as np
from Library import *
import json


dfDictist,indexes_names=load_indexes("Data/indexes")
gamma=0.5
R=compute_Returns(dfDictist,indexes_names,r_critic,"../Data/cleanData/TestData.json",indexes_names,gamma)
print(R.head)