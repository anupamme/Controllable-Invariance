import subprocess

def available_gpu(setting, num_g, p_per_gpu, black_gpu_set=None):
    running = []
    for i in range(num_g):
        running += [0]
    if setting == "rabat":
        proc = subprocess.Popen(['nvidia-smi'],stdout=subprocess.PIPE)
        content = proc.stdout.readlines()
        for k in range(21, len(content) - 1):
            if "/usr/lib/xorg/Xorg" in content[k]:
                continue
            g = int(content[k][1: 7])
            if g == 2 or g == 0:
                g = 2 - g
            if g < num_g:
                running[g] += 1
        for i in range(num_g):
            if running[i] < p_per_gpu and (black_gpu_set is None or i not in black_gpu_set):
                return i
    elif setting == "tir":
        proc = subprocess.Popen(['squeue'],stdout=subprocess.PIPE)
        content_list = proc.stdout.readlines()
        content = "\n".join(content_list)
        current_job = content.count("qizhex")
        if current_job >= num_g:
            return None
        #if len(content_list) > 25:
        #    return None
        if content.find("PD") != -1 and content.find("qizhex") != -1:
            return None
        return 0
    else:
        assert False
    return None

if __name__ == "__main__":
    print available_gpu("tir", 2, 2)
