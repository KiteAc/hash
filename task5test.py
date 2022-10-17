import numpy as np

label = list({1, 2, 3, 4, 5, 5, 6, 2})
print(set(label))

c = np.random.choice(label, 1, replace= False)
print(c)