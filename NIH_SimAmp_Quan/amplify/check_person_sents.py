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

source = "../result/IBM_no-noise.csv"
df = pd.read_csv(source)
df_filtered = df[df.iloc[:, 0].str.split("_").str[4]=="pd06"]
df_filtered = df_filtered.sort_values(by=df_filtered.columns[0]) 
df_filtered.to_csv('pd06_amp_vs_na.csv', index=False)
