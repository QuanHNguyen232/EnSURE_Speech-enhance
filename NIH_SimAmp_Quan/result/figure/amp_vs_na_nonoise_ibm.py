import pandas as pd
import matplotlib.pyplot as plt

# sim_NM_amp_nn_oc11_h15_2  OR   sim_NM_na_nn_pd16_h2_10
# NM: No mask
# amp: with amplifier
# nn: no noise
# oc11: people id
# h15_2: sentence id

df = pd.read_csv("../MS_no-noise.csv")
pd_oc = df["wavfile"].str.split("_", expand=True)[2]
is_amplified = pd_oc.str.contains('amp', case=False)
is_noamp = pd_oc.str.contains('na', case=False)

amplified_df = df[is_amplified]
noamp_df = df[is_noamp]
amplified_wer = amplified_df["wer"]
amplified_cer = amplified_df["cer"]

noamp_wer = noamp_df["wer"]
noamp_cer = noamp_df["cer"]

fig, ax = plt.subplots()
ax.boxplot([amplified_wer, noamp_wer, amplified_cer, noamp_cer], labels=['Amplified WER', 'No-Amp WER', 'Amplified CER', 'No-Amp CER'])
ax.set_ylabel('WER and CER')
ax.set_title('Comparison of WER and CER between Amplified and No-Amplified for MS API, No noise condition')
plt.savefig('MS_no-noise.png')