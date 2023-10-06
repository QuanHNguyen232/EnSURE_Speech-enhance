## Note to run this project
### Project goal: Amplify patients speech to improve the ASR accuracy
File Description:

All sound source are pre-recorded from PD and OC people. Then they replay those sound by __Head&Torso with mouth simulator__, with different noise conditions, amplication status, mask status. 

```data/sim_production_audio_rescaled_70dB:``` The sound source is rescaled to 70dB. With **no noise**, **close to recorder.** 
(Q:record distance? The sound source is amplified or not?)

sim_oc02_habitual_nm_h17_3_ch1_70db.wav

```oc02: healthy old people id=2``` 

```nm: no mask``` 

```h17_3: setence id``` 

The following folder are recorded by **2 meter distance**
```data/no-noise:``` no noise.

```data/babble:``` 5dB SNR babble noise.

sim_NM_amp_nn_oc02_h17_1

```NM: no mask```

```amp: amplified by MiniBuddy Personal Voice Amplifier```

```na: not amplified```


Amplify Setting 1: 
Only amplify energy 1-3kHz


Amplify setting 2: 
Only amplify enery 1-8kHz

Step1: For no noise condition, amplify different strategies. 

Compare if our amplified with the original amplifed for PD patients. Specifically, choose 
```no-noise/sim_NM_[na]_nn_oc02_h17_1.wav```. Only choose the no amplified audio to amplify. Then save it at same name to the amplified dir. For example, if we amplify it with 1k-3k with factor as 2, then the file are saved to ```1000_3000_2/sim_NM_[na]_nn_oc02_h17_1.wav```

### Step1:
Run ```amplify/amplify_by_freq.py``` to get amplified audio under ```amplify```folder.
### Step2:
Run ASR API to recognize the audios.
Run ```ASRs/IBM_API.py``` with changing the source and output folder. Then produce a transcriptions of all audios in .txt. For example, input folder is ```./amplify/1000_3000_2/*.wav```, output is ```IBM_nn_1000_3000_2.txt```
### Step3:
Draw result based on the transcriptions
Run ```amplify/compare.py```

## TODO:
1. Check if only specific user is bad. 
    * All PDs are usually worse than OC. Check ```amplify/IBM WER Per person.png```
    * Minibuddy amplifier can improve performance. Check ```pd06_amp_vs_na.csv```
2. Use diffusion model (Implement DiffWave)
    * check audioPure code

3. Check speech enhancement paper

4. What's the difference between normal and pd?

Find specific flaws.
Amplitude? 频率缺失？ 不连贯？ inconsistancy.

正常人train diffwave. 

Inference pds

Sample to sample. 连词起来。 

---

*Note*: use torch1.7 if indicated in the files, amplify by freq/formant files must use DL conda environment.

HAVE NOT READ YET:
* [Comparative analysis of Dysarthric speech recognition: multiple features and robust templates](https://link.springer.com/article/10.1007/s11042-022-12937-6)
* [Automatic Speech Recognition with Deep Neural Networks for Impaired Speech](https://link.springer.com/chapter/10.1007/978-3-319-49169-1_10)
* [Two-Step Acoustic Model Adaptation for Dysarthric Speech Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053725&tag=1)
* [Automatic speech recognition and a review of its functioning with dysarthric speech](https://www.tandfonline.com/doi/abs/10.1080/07434610012331278904)
* [Speech Vision: An End-to-End Deep Learning-Based Dysarthric Automatic Speech Recognition System](https://ieeexplore.ieee.org/abstract/document/9419963)

---
My pipeline will be:
1. Speech to text:
    1. Pre-process: amplification (by formants/freq) as add-ons method
    1. Fine-tune an ASR model (like [whisper](https://huggingface.co/openai/whisper-medium) -> [finetune-tutorial](https://huggingface.co/blog/fine-tune-whisper))
    1. Post-process: correct wrong words based on the sentence context using some type of pretrained models or fine-tune them (optional)
1. Text to speech:
    * Use pretrain Tacotron

It should be real-time inference (focus on speed) -> speech to speech. Consider Speech Enhancement task:
1. [SpeechBrain](https://github.com/speechbrain/speechbrain). HuggingFace models for Speech Enhancement:
    * [speechbrain/sepformer-whamr-enhancement](https://huggingface.co/speechbrain/sepformer-whamr-enhancement)
    * [speechbrain/metricgan-plus-voicebank](https://huggingface.co/speechbrain/metricgan-plus-voicebank)
    * Tutorial (try to combine those 2 into fine-tune pipeline for our task):
        * Speech Enhancement from scratch [tutorial](https://speechbrain.github.io/tutorial_enhancement.html)
        * ASR [fine-tune](https://speechbrain.github.io/tutorial_advanced.html#:~:text=Pre%2Dtrained%20Models%20and%20Fine%2DTuning%20with) tutorial

1. Use [SEGAN-pytorch](https://github.com/santi-pdp/segan_pytorch) combining with Phoneme Recognition:
    * Fine-tuning Wav2Vec2 for phoneme recognition [github](https://github.com/kosuke-kitahara/xlsr-wav2vec2-phoneme-recognition/blob/main/Fine_tuning_XLSR_Wav2Vec2_for_Phoneme_Recognition.ipynb). Info about Wav2Vec2Phoneme on [huggingface](https://huggingface.co/docs/transformers/model_doc/wav2vec2_phoneme).