import os
import codecs
import string


def transform(inf, ouf):
    inp = codecs.open(inf, "r", "utf-8")
    oup = codecs.open(ouf, "w", "utf-8")
    print inf
    if inf.find("train.pt") != -1:
        return
    for line in inp.readlines():
        st = []
        line = line.strip()
        for word in line.split():
            if 'a' <= word[0] <= 'z' or 'A' <= word[0] <= 'Z' or word[0] in string.punctuation:
                st += [word]
            else:
                for c in word:
                    #if inf.find(".zh") != -1:
                    #    print c
                    st += [c]
        oup.write(" ".join(st) + "\n")
    oup.close()

if __name__ == "__main__":
    indir = "../../../data_MT/iwslt15_en_zh"
    for in_f in os.listdir(indir):
        transform(indir + "/" + in_f, indir + "_c/" + in_f)
