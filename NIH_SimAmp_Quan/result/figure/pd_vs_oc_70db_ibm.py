import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../MS_70db.csv")
pd_oc = df["wavfile"].str.split("_", expand=True)[1]
is_patient = pd_oc.str.contains('pd', case=False)
is_normal = pd_oc.str.contains('oc', case=False)

patients_df = df[is_patient]
normal_df = df[is_normal]
patients_wer = patients_df["wer"]
patients_cer = patients_df["cer"]

normal_wer = normal_df["wer"]
normal_cer = normal_df["cer"]

fig, ax = plt.subplots()
ax.boxplot([patients_wer, normal_wer, patients_cer, normal_cer], labels=['Patients WER', 'Normal WER', 'Patients CER', 'Normal CER'])
ax.set_ylabel('WER and CER')
ax.set_title('Comparison of WER and CER between Patients and Normal People for MicroSoft API')
plt.savefig('MS_70dB.png')