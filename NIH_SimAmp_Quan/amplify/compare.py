#######################
# Author: Hanqing
# Date: 05/15/2023
# Description: This code is to compare the ASR WER, CER difference between 
# four amplify settings. 
#    1. Only amplify 1-3k; [na]+1-3k
#    2. Only amplify 1-8k; [na]+1-8k 
#    3. Amplify MiniBuddy Personal Voice Amplifier [amp]
#    4. No amp [na]
#######################
import pandas as pd
from jiwer import wer, cer
import matplotlib.pyplot as plt
import seaborn as sns


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
    gts = []
    preds = []
    wers = []
    cers = []
    with open(asr_data, "r") as f:
        ASR_preds = f.readlines()
        for idx, ASR_pred in enumerate(ASR_preds):

            print("{}/{}".format(idx, len(ASR_preds)))
            wavfile_ = ASR_pred.split(":")[0]  # audio path
            amplified = wavfile_.split("_")[2]  # if it is "amp" or "na"
            if select_amp==False and amplified == "na":  # skip "amp" sample, only choose "na" sound
                if API_name == "IBM":  # Becasue the IBM, Google result has no ending dot
                    pred_text = ASR_pred.split(":")[1][:-2] + "."
                elif API_name == "Google":  # Becasue the IBM, Google result has no ending dot
                    pred_text = ASR_pred.split(":")[1][:-1] + "."
                else:
                    pred_text = ASR_pred.split(":")[1]
                gt_text = get_gt(df, wavfile_)
                gt_text = gt_text.lower()
                pred_text = pred_text.lower()
                _wer = wer(gt_text, pred_text)
                _cer = cer(gt_text, pred_text)
                wavefiles.append(wavfile_)
                gts.append(gt_text)
                preds.append(pred_text)
                wers.append(_wer)
                cers.append(_cer)
            if select_amp==True and amplified == "amp":  # skip "amp" sample, only choose "na" sound
                if API_name == "IBM" or "Google":  # Becasue the IBM, Google result has no ending dot
                    pred_text = ASR_pred.split(":")[1][:-2] + "."
                else:
                    pred_text = ASR_pred.split(":")[1]
                gt_text = get_gt(df, wavfile_)
                gt_text = gt_text.lower()
                pred_text = pred_text.lower()
                _wer = wer(gt_text, pred_text)
                _cer = cer(gt_text, pred_text)
                wavefiles.append(wavfile_)
                gts.append(gt_text)
                preds.append(pred_text)
                wers.append(_wer)
                cers.append(_cer)
    return wers, cers


if __name__ == '__main__':
    _1_3_k_wer, _1_3_k_cer = get_metric("IBM_nn_1000_3000_2.txt")
    _1_8_k_wer, _1_8_k_cer = get_metric("IBM_nn_1000_8000_2.txt")
    minbuddy_amp_wer, minibuddy_amp_cer = get_metric("IBM_no-noise.txt", select_amp=True)
    no_amp_wer, no_amp_cer = get_metric("IBM_no-noise.txt")
    no_amp_diffwave_wer, no_amp_diffwave_cer = get_metric("IBM_nn_diffwave.txt")
    amp_formant_wer, amp_formant_cer = get_metric("IBM_nn_amplify_formant.txt")

    data = [_1_3_k_wer, _1_8_k_wer, minbuddy_amp_wer, amp_formant_wer, no_amp_wer, no_amp_diffwave_wer]
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=data)
    ax.set_xticklabels(['Amp 1-3k', 'Amp 1-8k', 'Amp_minibuddy', 'Amp_formant', 'No_amp', 'No_amp_diffwave'])
    plt.title("IBM WER")
    plt.savefig("IBM_WER_formant.png")



    fig = plt.figure(figsize=(10, 6))
    data = [_1_3_k_cer, _1_8_k_cer, minibuddy_amp_cer, amp_formant_cer, no_amp_cer, no_amp_diffwave_cer]
    ax = sns.boxplot(data=data)
    ax.set_xticklabels(['Amp 1-3k', 'Amp 1-8k', 'Amp_minibody', 'Amp_formant', 'No_amp', 'No_amp_diffwave'])
    plt.title("IBM CER")
    plt.savefig("IBM_CER_formant.png")




    

    

