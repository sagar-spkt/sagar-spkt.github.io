---
title: 'Critical Analysis: Why do tree-based models still outperform deep learning on typical tabular data?'
date: 2023-06-11
permalink: /posts/2023/06/tree-better-than-deeplearning/
excerpt: "It is an annotated criticism on the paper `Why do tree-based models still outperform deep learning on typical tabular data?`. A personal view on why it was accepted for publication at NEURIPS 2022 is discussed along with a brief description of the key contributions and the significance of the findings presented in the paper. Also, some of the key strengths and weaknesses with this paper, the approach taken by the authors, and the clarity and ease of understanding of the writing are discussed herewith."
comments: true
tags:
  - Paper Reading
  - Critics
  - Machine Learning
  - Deeplearning
  - Tree-Based Models
---

In recent years, the machine learning community has witnessed significant advancements in deep learning models. However, a perplexing phenomenon encountered by industrial machine learning practitioners is that even the simplest tree-based models often outperform advanced deep learning models on real-world projects involving tabular data. In this blog post, I will delve into the critical analysis of a research paper that sheds light on this issue and provides valuable insights for practitioners working with tabular data.

The paper "Why do tree-based models still outperform deep learning on typical tabular data?" by Grinsztajn et al. [2023], establishes a new standard set of datasets with clear characteristics of tabular data and benchmarks various tree-based and deep learning models on these datasets. The results debunk common myths about the performance of neural networks on tabular data, highlighting the importance of understanding inductive biases and the impact of uninformative features and irregular functions on model performance.

By discussing the strengths and weaknesses of the paper, I aim to provide a comprehensive understanding of why tree-based models continue to outshine deep learning models on typical tabular data. This analysis will be particularly useful for industrial machine learning practitioners who are often puzzled by the seemingly inferior performance of deep learning models on tabular data. Also, some of the key strengths and weaknesses with this paper, the approach taken by the authors, and the clarity and ease of understanding of the writing are discussed herewith.

In the PDF attached below in the blog post, I have included annotations for the critical analysis . These annotations were made while I was reading the paper.


### Key Contributions and Their Significances:
- The paper establishes a new standard set of datasets with clear characteristics of tabular data. The authors also provided precise processing methods used to create them. They claim that having such homogeneous datasets allows researchers to investigate inductive biases purely suited for tabular data.
- The authors benchmarked standard tree-based models(RF, GBTs, XGBoost) popular among practitioners and SOTA deep learning models for tabular data(MLP, ResNet, FT Transformer, SAINT) on those datasets. While doing so, the authors took into consideration different hyperparameter optimization budgets. The variance introduced due to it is addressed intelligently by shuffling the random search order multiple times. The result debunked two myths and pointed out that hyperparameter tuning doesn’t make neural nets state-of-the-art, and categorical variables are not the main weakness of neural networks.
- While investigating why tree-based models outperform deep learning models by transforming data to alter their performance gap, the authors shed light on their different inductive biases. These findings are of significant importance as they guide future research to make tabular-specific neural networks robust to uninformative features, deal with irregular functions, and be rotationally non-invariant in a computationally cheaper way.

### Strengths:
- Though not for all experiment settings, the author provides sound justification for some of their choices. For instance, Bayesian optimization was not chosen over random search as it doesn’t allow reshuffling of the search order, and their ablation study also shows it doesn’t provide a significant improvement over random search. Also, their choices of data preparation steps cohere with the goal of making homogeneous datasets.
- The conclusions drawn are backed by complementary empirical evidence. For instance, while gauging the effects of uninformative features, authors draw the same conclusion by both adding and removing uninformative features or by training on informative and uninformative features separately. Similarly, their empirical conclusion about the link between rotational invariance and uninformative features is validated by the theoretical link provided by Ng [2004].
- The paper is well-sectioned with pertinent information. The introduction covers all the important aspects of the whole paper. It is an easy-to-follow paper. The codebase is also available publicly.


### Weaknesses:
- The authors claim they provide new comprehensive datasets for a standard benchmark. However, the criteria used while creating these datasets ignore many features of real-world datasets, questioning their usability in standard benchmarking.
- Kadra et al. [2021a] uses a “cocktail” of regularization on MLPs and get competitive with XGBoost on a similar random search budget. Rather than speculating the performance was particularly due to the presence of “deterministic” datasets, the authors could’ve proven it empirically by measuring the performance of MLP with regularization on their newly created datasets.
- Although I anticipate conclusions drawn in the paper hold for small and large datasets, experiments only with medium-sized datasets leave a place for doubt.
- I think techniques to remove side issues contradict some criteria mentioned in 3.1, such as “Not too easy,” and “real-world data.” 
- No clear explanation is given why multi-class tasks are binarised, why only the top 5 features based on RF importance ranking were taken to study the impact of irregular functions, why the search order was shuffled within a single random search run instead of considering a new one, and why “ReduceOnPlateau” LR Scheduler was chosen for MLP.


### Summary:
Overall, the significance of the contributions and strengths of the paper beats its weaknesses. That is why I think it was accepted in NeurIPS 2022. Also, this paper provides explanations for practitioners perplexed by the inferior performance of deep learning models on tabular data.

### Annotated Paper:
<iframe src="https://drive.google.com/file/d/1cjwxq9xzFK2MJ_wMUO5b1m3ejbln2lJ3/preview" style="width:100%; height:100vh;"></iframe>

### References:
- LéoGrinsztajn,EdouardOyallon,andGaëlVaroquaux.“Whydotree-basedmodelsstilloutperform deeplearningontabulardata?”(July2022).arXiv: 2207.08815[cs.LG]
- Andrew Y. Ng. Feature selection, L 1 vs. L 2 regularization, and rotational invariance. In Twenty-First International Conference on Machine Learning - ICML ’04, page 78, Banff, Alberta, Canada, 2004. ACM Press. doi: 10.1145/1015330.1015435.
- Arlind Kadra, Marius Lindauer, Frank Hutter, and Josif Grabocka. Well-tuned Simple Nets Excel on
Tabular Datasets, November 2021a.
 