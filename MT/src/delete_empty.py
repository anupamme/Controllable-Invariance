import os
import shutil
import time
import datetime

def main():
    obj_dir = "../obj/"
    for sub_dir in os.listdir(obj_dir):
        d = obj_dir + sub_dir
        if not os.path.isdir(d):
            continue
        files = " ".join(os.listdir(d))
        modify_time = os.path.getmtime(d)
        modify_time_elapse = (time.time() - modify_time) / 3600. 
        modify_date = datetime.date.fromtimestamp(modify_time).day
        today = datetime.date.today().day
        if "npz" not in files and "model_e13" not in files and modify_time_elapse > 5:
            shutil.rmtree(d)

if __name__ == "__main__":
    main()
