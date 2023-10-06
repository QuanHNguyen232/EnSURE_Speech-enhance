import pandas as pd
import os
import glob
from collections import defaultdict

if __name__ == '__main__':
    file_list = glob.glob('data/no-noise/*.wav')
    pd_file_list = [filename for filename in file_list if 'oc' not in filename and 'pd' in filename]
    print(len(pd_file_list))
    mydict = defaultdict(set)
    for filename in pd_file_list:
        name = os.path.basename(filename).split('.')[:-1][0]
        if 'enhance' in name: continue
        try:
            name = name.split('_')
            sent_id = name[-1]
            list_id = name[-2][1:]
            mydict[int(list_id)].add(int(sent_id))
            if int(list_id)==0: print(name)
        except Exception as e:
            print(e, name)
    
    for key in mydict.keys():
        print(key, mydict[key])