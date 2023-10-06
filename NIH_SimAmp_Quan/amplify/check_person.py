#######################
# Author: Hanqing
# Date: 05/15/2023
# Description: This code is to check wer by person

#######################
import pandas as pd
from jiwer import wer, cer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_gt(df, wavfile):
    # wavfile = "sim_oc11_habitual_nm_h15_10_ch1_70db"
    # wavfile = "sim_NM_amp_nn_oc02_h17_1.wav"  no noise
    # wavfile = "sim_NM_amp_babble_oc02_h17_1.wav" babble
    if wavfile[:-4]=="70db":
        list_id = wavfile.split("_")[4]  # h15
        list_id = list_id[1:]  # 15
        sent_id = wavfile.split("_")[5]
        list_results = df.loc[df['LIST'] == int(list_id)]
        gt_text = list_results.iloc[int(sent_id) - 1]['SENTENCE'][3:].lstrip()
    else:
        list_id = wavfile.split("_")[5]  # h15
        list_id = list_id[1:]  # 15
        sent_id = wavfile.split("_")[6]
        list_results = df.loc[df['LIST'] == int(list_id)]
        gt_text = list_results.iloc[int(sent_id) - 1]['SENTENCE'][3:].lstrip()
    return gt_text

########################################
# Only PD, only na
########################################
def get_metric(asr_data, select_amp=False):
    API_name = asr_data.split("/")[-1].split("_")[0]
    df = pd.read_excel("../Harvard_Sentences_Keywords_CASALab_homophones.xlsx")
    wavefiles = []
    wers = {}
    cers = {}
    with open(asr_data, "r") as f:
        ASR_preds = f.readlines()
        for idx, ASR_pred in enumerate(ASR_preds):

            print("{}/{}".format(idx, len(ASR_preds)))
            wavfile_ = ASR_pred.split(":")[0]  # audio path
            amplified = wavfile_.split("_")[2]  # if it is "amp" or "na"
            spkid = wavfile_.split("_")[4]
            if (select_amp==False and amplified == "na") or (select_amp==True and amplified == "amp"):
                # Non-amplified                              "amp" for minibuddy
                if API_name == "IBM" or "Google":  
                    pred_text = ASR_pred.split(":")[1][:-2] + "."
                else:
                    pred_text = ASR_pred.split(":")[1]
                gt_text = get_gt(df, wavfile_)
                gt_text = gt_text.lower()
                pred_text = pred_text.lower()
                _wer = wer(gt_text, pred_text)
                _cer = cer(gt_text, pred_text)
                if spkid not in wers.keys():
                    wers[spkid] = []
                    cers[spkid] = []
                wers[spkid].append(_wer)
                cers[spkid].append(_cer)
    return wers, cers


if __name__ == '__main__':
    no_amp_wer, no_amp_cer = get_metric("IBM_no-noise.txt", select_amp=False)
    no_amp_wer = dict(sorted(no_amp_wer.items()))
    no_amp_cer = dict(sorted(no_amp_cer.items()))


    minibuddy_amp_wer, minibuddy_amp_cer = get_metric("IBM_no-noise.txt", select_amp=True)
    minibuddy_amp_wer = dict(sorted(minibuddy_amp_wer.items()))
    minibuddy_amp_cer = dict(sorted(minibuddy_amp_cer.items()))

    N = len(no_amp_wer.keys())
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35          # the width of the bars
    
    fig, ax = plt.subplots(figsize=(16,9))
    bp1 = ax.boxplot(no_amp_wer.values(), positions=ind - width/2, widths=width, patch_artist=True, boxprops=dict(facecolor="C0"))
    bp2 = ax.boxplot(minibuddy_amp_wer.values(), positions=ind + width/2, widths=width, patch_artist=True, boxprops=dict(facecolor="C1"))

    ax.set_xticks(ind)
    ax.set_xticklabels(no_amp_wer.keys())
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['No amp', 'Minibuddy'], loc='upper right')

    # bp = ax.boxplot(no_amp_wer.values())
    # ax.set_xticklabels(no_amp_wer.keys())

    plt.xlabel('Speakers ID')
    # plt.ylabel('Word Error Rate')
    plt.title("IBM WER")

    plt.savefig('IBM WER Per person.png')

    plt.show()
