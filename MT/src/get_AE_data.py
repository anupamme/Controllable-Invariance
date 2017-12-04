#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re, os

def get_raw():
    inp = open("/projects/tir1/users/qizhex/workspace/simp/data_MT/all_talks.tsv", "r")
    AE_dir = "/projects/tir1/users/qizhex/workspace/simp/data_Multi-MT/AE/orig"
    title = inp.readline().strip()
    languages = title.split("\t")
    position = {}
    for i, lang in enumerate(languages):
        position[lang] = i
    language_interested = ["en", "fr", "de"]
    AE_language_file = []
    for lang in language_interested:
        AE_language_file.append(open(AE_dir + "/%s.txt" % lang, "w"))
    for whole_st in inp.readlines():
        line = whole_st.strip().split("\t")
        if len(line) != len(languages):
            print line
        else:
            for i, lang in enumerate(language_interested):
                p = position[lang]
                st = line[p].strip()
                '''st = st.replace("(Gelächter)", "")
                st = st.replace("(Applaus)", "")
                st = st.replace("(Applause)", "")
                st = st.replace("(Laughter)", "")
                st = st.replace("(Applaudissements)", "")
                st = st.replace("(Rires)", "")'''
                st = st.replace("\"\"\"\"", "\"")
                st = re.sub(r'\([^)]*\)', '', st)
                st = st.strip()
                if st.find("__NULL__") == -1 and st != "":
                    AE_language_file[i].write(st + "\n")
    for AE_file in AE_language_file:
        AE_file.close()

def filter_exist():
    data = "/projects/tir1/users/qizhex/workspace/simp/data_Multi-MT/AE"
    appeared = {}
    ori_dir = "/projects/tir1/users/qizhex/workspace/simp/data_Multi-MT/iwslt15_de_fr_en_bpe/orig"
    leng = 50
    for inp in os.listdir(ori_dir):
        if inp.find(".de") != -1 or inp.find(".en") != -1 or inp.find(".fr") != -1:
            inf = open(ori_dir + "/" + inp, "r")
            for st in inf:
                appeared[st.strip()[:leng]] = st
                appeared[st.strip()[-leng:]] = st
                appeared[st.strip()] = st
    for inp in os.listdir(data + "/tmp"):
        if inp.find(".txt") == -1:
            continue
        inf = open(data + "/tmp/%s" % inp, "r")
        ouf = open(data + "/prep/%s" % inp, "w")
        for st in inf:
            st = st.strip()
            if st.count(" ") < 50 and st not in appeared:
                if st[:leng] not in appeared and st[-leng:] not in appeared:
                    ouf.write(st + "\n")
                elif st[:leng] in appeared:
                    #print st
                    #print appeared[st[:leng]]
                    pass
                elif st[-leng:] in appeared:
                    #print st
                    #print appeared[st[-leng:]]
                    pass
                else:
                    assert False
        ouf.close()


if __name__ == "__main__":
    filter_exist()