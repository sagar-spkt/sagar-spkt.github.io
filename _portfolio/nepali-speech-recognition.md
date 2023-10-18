---
title: "Nepali Automatic Speech Recognition using Wav2Vec2 Model"
excerpt: "Worked on finetuning Wav2Vec2 model on OpenSLR’s Nepali ASR dataset. Preprocessed audio dataset with silence removal, voice activity detection, and noise augmentation."
collection: portfolio
date: 2022-08-23
---

Description
=====
This project involves the fine-tuning of XLSR-Wav2Vec2, a multilingual Automatic Speech Recognition (ASR) model, for the Nepali language. The XLSR model was developed by Facebook AI and stands for cross-lingual speech representations, indicating the model's ability to learn speech representations useful across multiple languages.

The project begins with data preprocessing and dataset preparation using the OpenSLR Nepali ASR dataset. This dataset includes fields such as "utterance_id", "speaker_id", "utterance", "transcription", and "num_frames". The dataset is then loaded into PyTorch for further processing.

The project utilizes the Wav2Vec2 feature extractor and tokenizer to process the audio files and transcriptions. A custom PyTorch dataset class, NepaliASRProcessedDataset, is created to process the audio files and transcriptions as they are requested, making it possible to work with the large dataset given memory constraints.

The project then splits the dataset into a training set (85%) and a test set (15%). This is done manually as the dataset does not contain separate splits for training and evaluation.

The project utilizes the XLSR model, which was pretrained on 128 languages and includes up to two billion parameters. The pretrained model is fine-tuned on the Nepali ASR dataset. The XLSR model learns contextualized speech representations by masking feature vectors before passing them to a transformer network. A single linear layer is added on top of the pretrained network for fine-tuning on labeled audio data for tasks such as speech recognition.

The project demonstrates that the XLSR model can be effectively fine-tuned for ASR tasks in languages other than those it was pretrained on, showcasing its versatility and the effectiveness of cross-lingual speech representations in multilingual ASR tasks.

Implementation Details
====
Please read my [blog post](https://spktsagar.com/posts/2022/08/finetune-xlsr-nepali/).

Presentation/Demo
====
<iframe
	src="https://spktsagar-spktsagar-wav2vec2-nepali-asr.hf.space"
	frameborder="0"
	width="100%"
	height="450"
></iframe>
