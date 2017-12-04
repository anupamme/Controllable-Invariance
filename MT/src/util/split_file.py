import os

if __name__ == "__main__":
    dataset_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../../data/%s/%s"
    suffix = ""
    dataset = "newsela"
    #dataset = "simple_wiki"
    source_file = dataset_dir % (dataset, "test.normal.tok%s" % suffix)
    target_file = dataset_dir % (dataset, "test.simple.tok%s" % suffix)
    source_file = open(source_file, "r")
    target_file = open(target_file, "r")
    aligned_sentences = {}
    sources = []
    for s, t in zip(source_file, target_file):
        s = s.strip()
        t = t.strip()
        if s not in aligned_sentences:
            aligned_sentences[s] = []
            sources.append(s)
        aligned_sentences[s].append(t)
    max_ref = 0
    test_dataset = dataset_dir % (dataset, "test")
    if not os.path.exists(test_dataset):
        os.mkdir(test_dataset)
    for s in sources:
        if len(aligned_sentences[s]) > max_ref:
            max_ref = len(aligned_sentences[s])
    refs = []
    for i in range(max_ref):
        refs += [open(os.path.join(test_dataset, "test.simple.%s" % str(i)), "w")]
    new_src_file = open(os.path.join(test_dataset, "test.normal"), "w")
    for s in sources:
        new_src_file.write(s + "\n")
        for j in range(max_ref):
            if j < len(aligned_sentences[s]):
                st = aligned_sentences[s][j]
            else:
                st = ""
            refs[j].write(st + "\n")
    for i in range(max_ref):
        refs[i].close()
