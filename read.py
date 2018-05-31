from utils import DATA_RAW_DIR, DATA_PROCESSED_DIR
import os
import random
import codecs

topic_type_path = os.path.join(DATA_RAW_DIR, 'topic_type.txt')

topic_list = []
key_set = []
with codecs.open(topic_type_path, 'r', 'utf-8') as fin:
    line = fin.readline()
    print("WWWW")
    while line:
        print("line", line)
        toks = line.strip().split('：')
        topic = toks[0]
        set_words = toks[1]
        # content = content.replace(' ', '')
        set_words = set_words.split('，')
        key_set.append(set_words)
        topic_list.append(topic)
        line = fin.readline()
str = "爱情"
c = topic_list.index(str)
print(c)
n = len(key_set[c])
s = key_set[c][random.randint(0, n - 1)]
print("s", s)
# for i in range(n):
#     b = key_set[c][i]
#     new_key_set.append(b)

# m = len(new_key_set)
# s = new_key_set[random.randint(0, m - 1)]
# return new_key_set, s
