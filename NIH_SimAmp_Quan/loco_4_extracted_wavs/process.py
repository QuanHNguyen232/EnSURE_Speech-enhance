import os
from collections import defaultdict
import shutil
import torchaudio
import subprocess
import pandas as pd
import string

def simple_clean_text(text):
    # lower case, remove punctuation
    return text.translate(str.maketrans('', '', string.punctuation)).lower().strip()

def get_gt():
    # get ground truth from csv file
    my_dict = defaultdict(dict)
    df = pd.read_excel('../Harvard_Sentences_Keywords_CASALab_homophones.xlsx')
    for idx, row in df.iterrows():
        list_id = int(row['LIST'])
        text = simple_clean_text(row['SENTENCE']).split()
        sent_id, *sent = simple_clean_text(row['SENTENCE']).split()
        my_dict[list_id][int(sent_id)] = ' '.join(sent)
    return my_dict
    
if __name__ == '__main__':
    
    # with open('./groundtruth_transcript.txt', 'r') as f:
    #     new_gt = f.readlines()
    # new_gt = [l.strip().split(',') for l in new_gt if 'PD' in l]
    # new_gt = {k:v for (k, v) in new_gt}
    # print(new_gt)


    # old_gt = {}
    # # GET SENTENCE
    # old_fs = os.listdir('./old_noisy')
    # gt_dict = get_gt()
    # for filename in old_fs:
    #     basename = os.path.basename(filename).replace('.wav', '')
    #     if "70db" in basename: # "sim_oc11_habitual_nm_h15_10_ch1_70db"
    #         list_id = basename.split("_")[4][1:]  # h15 -> 15
    #         sent_id = basename.split("_")[5]
    #     else: # "sim_NM_amp_babble_pd02_h17_1"
    #         list_id = basename.split("_")[5][1:]  # h17 -> 17
    #         sent_id = basename.split("_")[6]
    #     list_id, sent_id = int(list_id), int(sent_id)
    #     sent = gt_dict[list_id][sent_id]
    #     old_gt[filename] = sent
    
    # with open('labels.txt', 'w') as f:
    #     for k, v in old_gt.items():
    #         f.write(f'{k},{v}\n')
    #     for k, v in new_gt.items():
    #         f.write(f'{k}.wav,{v}\n')
    
    with open('./final_data/labels.txt', 'r') as f:
        lines = f.readlines()
        labels = [l.strip().split(',') for l in lines]
    labels = {k:v for k,v in labels}
    
    # print(labels)
    movefiles = [f for f in labels.keys() if 'sim' in f]
    print(len(movefiles))

    d = defaultdict(int)
    for filename in (movefiles):
        basename = os.path.basename(filename).replace('.wav', '')
        # GET SENTENCE
        if "70db" in basename: # "sim_oc11_habitual_nm_h15_10_ch1_70db"
            list_id = basename.split("_")[4][1:]  # h15 -> 15
            sent_id = basename.split("_")[5]
        else: # "sim_NM_amp_babble_pd02_h17_1"
            list_id = basename.split("_")[5][1:]  # h17 -> 17
            sent_id = basename.split("_")[6]
        list_id, sent_id = int(list_id), int(sent_id)
        d[list_id] += 1
    print(d)