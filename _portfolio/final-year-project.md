---
title: "Voice Cloning using Transfer Learning from Speaker Verification to Multispeaker TTS"
excerpt: "Worked on implementing Text Independent Speaker Verification in order to condition the Multispeaker version of Tacotron to synthesize speech of text on the voice of the given person in the wild. Implemented sequence-to-sequence encoder-decoder model based on attention mechanism."
collection: portfolio
date: 2020-01-23
---

Abstract
=====
A typical text-to-speech (TTS) synthesis contains multiple stages consisting of frontend, the task associated with the text, and backend, tasks associated with speech. Such systems require extensive domain knowledge and top-level engineering designs thus making it dependent on human inputs and vulnerable to human errors. We use a deep neural network-based system for text-to-speech synthesis that is able to generate speech audio in the voice of different speakers, including those unseen during training. We are using Google’s Tacotron based system for this voice cloning purpose. Our system consists of three independent components where one neural network model i.e. Speaker Encoder, that gives speaker embeddings, is trained independently using Generalized End-to-End Loss function for speaker verification task. The second neural network model (i.e. a Sequential Synthesis Network, that estimates linear spectrograms) and the final component (i.e. Vocoder, a post-processor that converts the linear spectrograms into Mel-spectrograms and finally generating speech) are combined together and trained separately using transfer learning in which we freeze the weights of Speaker Encoder. This complete model is a zero-shot generative model that can clone speaker's voice with just a few seconds of utterance. 

Github Repo
====
[![](/images/portfolio/final_project_git_preview.png)](https://github.com/sagar-spkt/SV2MTTS)

Presentation
====
<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vSv4SixBC3ckb9hdkWFAS__FfdARxDYnUrSHvOkAn-f9u5u-e26NrCIEyklmh57fe99fKdRfGFASPa1/embed?start=true&loop=true&delayms=3000" frameborder="0" style="width:100%; height:50vh;" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>


Full Report
====
<iframe src="https://drive.google.com/file/d/1KdKuKxphEmgOsLkMK3mQGvxfnAlBtE2c/preview" style="width:100%; height:100vh;"></iframe>
