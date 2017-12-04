import os
import html
import codecs
import HTMLParser

parser = HTMLParser.HTMLParser()

def work_mul(data_dir, st):
    n = 8
    inf = "%s.8turkers.tok." % (data_dir + st)
    norm_file = codecs.open(inf + "norm", "r", "utf-8")
    label_file = []
    for i in range(n):
        label_file.append(codecs.open(inf + "turk.%d" % i, "r", "utf-8"))
    src = []
    tgt = []
    while True:
        st = norm_file.readline().strip()
        if st == "":
            break
        for i in range(n):
            src.append(st)
            tgt.append(label_file[i].readline().strip())
    print len(src)
    return src, tgt


def read_file(inf):
    inp = codecs.open(inf, "r", "utf-8")
    content = [parser.unescape(st.strip().lower()) for st in inp]
    return content

def write_file(ouf, data):
    oup = codecs.open(ouf, "w", "utf-8")
    for d in data:
        oup.write(d + "\n")
    oup.close()

def work_PWKP(data_dir, st):
    src = read_file(data_dir + "%s.normal" % st)
    tgt = read_file(data_dir + "%s.simple" % st)
    return src, tgt

def dump(data_dir, set_name, src, tgt):
    write_file(data_dir + set_name + ".normal.tok", src)
    write_file(data_dir + set_name + ".simple.tok", tgt)

if __name__ == "__main__":
    obj_dir = "../../data/"
    wiki_mul_dir = obj_dir + "wiki_mul/devtest/"
    test_src, test_tgt = work_mul(wiki_mul_dir, "test")
    dev_src, dev_tgt = work_mul(wiki_mul_dir, "tune")
    src_dict = {}
    for src in test_src + dev_src:
        src_dict[src] = 1
    PWKP = obj_dir + "PWKP/"
    train_src, train_tgt = work_PWKP(PWKP, "train")
    dev_src, dev_tgt = work_PWKP(PWKP, "dev")
    test = work_PWKP(PWKP, "test")
    train_src += dev_src
    train_tgt += dev_tgt
    filtered_src = []
    filtered_tgt = []
    dup_cnt = 0
    for src, tgt in zip(train_src, train_tgt):
        if src not in src_dict:
            filtered_src.append(src)
            filtered_tgt.append(tgt)
        else:
            dup_cnt += 1
    dump(obj_dir + "simple_wiki/", "test", test_src, test_tgt)
    dump(obj_dir + "simple_wiki/", "dev", dev_src, dev_tgt)
    dump(obj_dir + "simple_wiki/", "train", filtered_src, filtered_tgt)
    unlabel = read_file(obj_dir + "AE/wiki_unfiltered.txt")
    unlabel_filtered = [sent for sent in unlabel if sent not in src_dict]
    write_file(obj_dir + "AE/wiki.txt", unlabel_filtered)