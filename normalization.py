from sys import argv
from collections import Counter

datafile = open(argv[1])
data = list(map(lambda x : x.split(",")[0:-1], datafile.readlines()))
for x in data : x.pop(6)


print(data[0])
aux = [i[5] for i in data[1:]]
print(len(Counter(aux).keys()))