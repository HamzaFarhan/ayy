from itertools import groupby

for k, v in groupby([1, 1, 2, 3, 2, 1]):
    print(k)
    print(list(v))
