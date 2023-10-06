# conda activate torch1.7
# Give a ASR transcribe, return the metrics to a csv file
# ASR transcribe: [IBM]_[no-noise].txt
# metric file: [IBM]_[no-noise].csv
import pandas as pd
from jiwer import wer, cer
from pesq import pesq
from scipy.io import wavfile
from scipy.io import wavfile
import mir_eval


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


def get_audio_quality(ref, est):
    # fs, ref = wavfile.read(ref_file)
    # _, est = wavfile.read(est_file)

    length = min(len(ref), len(est))
    # compute all the metrics
    (sdr, sir, sar, _) = mir_eval.separation.bss_eval_sources(ref[:length], est[:length], compute_permutation=True)
    return sdr, sir, sar

def get_metric(asr_data):
    API_name = asr_data.split("/")[-1].split("_")[0]
    df = pd.read_excel("../Harvard_Sentences_Keywords_CASALab_homophones.xlsx")
    wavefiles = []
    gts = []
    preds = []
    wers = []
    cers = []
    sdrs = []
    sirs = []
    sars = []
    from scipy.io import wavfile
    ref_file = "../data/sim_production_audio_rescaled_70dB/sim_oc02_habitual_nm_h17_1_ch1_70db.wav"
    rate, ref = wavfile.read(ref_file)  # reference audio

    with open(asr_data, "r") as f:
        ASR_preds = f.readlines()
        for idx, ASR_pred in enumerate(ASR_preds):
            print("{}/{}".format(idx, 404))
            wavfile_ = ASR_pred.split(":")[0]  # audio path
            if API_name == "IBM" or "Google":  # Becasue the IBM, Google result has no ending dot
                pred_text = ASR_pred.split(":")[1][:-2] + "."
            else:
                pred_text = ASR_pred.split(":")[1]
            gt_text = get_gt(df, wavfile_)
            _wer = wer(gt_text, pred_text)
            _cer = cer(gt_text, pred_text)
            wavefiles.append(wavfile_)
            gts.append(gt_text)
            preds.append(pred_text)
            audio_folder = asr_data.split("_")[2][:-4]  # can be "no-noise, babble, 70dB"
            # print(audio_folder)
            audio_path = "../data/"+audio_folder+"/"+wavfile_+".wav"
            rate, est = wavfile.read(audio_path)
            sdr, sir, sar = get_audio_quality(ref, est)
            sdrs.append(sdr)
            sirs.append(sir)
            sars.append(sar)

            wers.append(_wer)
            cers.append(_cer)
    data = {"wavfile": wavefiles,
            "ground_truth": gts,
            "ASR_recognition": preds,
            "wer": wers,
            "cer": cers,
            "sdr": sdrs,
            "sir": sirs,
            "sar": sars
            }
    df_result = pd.DataFrame(data)
    csv_name = asr_data.split("/")[-1][:-4]
    df_result.to_csv(csv_name + ".csv", index=False)


if __name__ == '__main__':
    get_metric("../ASRs/ASR_Trans/IBM_babble.txt")
