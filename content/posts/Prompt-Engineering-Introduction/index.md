---
title: "Learning the deep learning for Earth System Research"
description: "It can feel overwhelming to learn new methodologies to apply to research. PNNL recently released the contents of a bootcamp on artificial intelligence (AI) and machine learning (ML) for their earth scientists, which may also help you get started!"
summary: "Seven modules were released by PNNL to demonstrate a broad display of AI/ML application to the Earth Sciences, a usefull tool for scientists looking to apply more AI/ML methodologies in their research."
date: 2026-02-13 # if you set date in the future, it will not show in the blog. Put past date during writing post
tags: ["Earth Science", "Bootcamp","Machine Learning","Education","python","neural networks","random forrest", "deep learning", "satellite imagery","hydrology", "PM2.5","ENSO forecasting"]
author: ["Goldberger, Lexie"]
series: ["AI-ML"]
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: true
#disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: false
ShowRssButtonInSectionTermList: true
UseHugoToc: true
math: true

cover:
    #image: "https://bestarion.com/wp-content/uploads/2024/06/prompt-engineering.jpg" # image path/url
    image: "https://i.ibb.co/jkyXKbSg/Picture8.png"
    caption: "AI has a multitude of uses in earth science research once you know the tools. Image generated using AI using stock imagery inspired by the boot camp. Credit: Nathan Johnson/ PNNL, Shutterstock.com"
    relative: false # when using page bundles set this to true
    hidden: false          # don't hide globally
    hiddenInList: true     # hide in list pages
    hiddenInSingle: false  # show inside post
editPost:
    URL: "https://github.com/ess-community/aiml-blogs/blob/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

## Introduction

The goal of the bootcamp was to help participants become familiar with the full range of AI/ML methods applicable to their research and understand the importance of data curation for AI/ML. Each lesson was divided into understanding the method and hands-on activity portions, with each hands-on activity featuring an Earth science-relevant domain application of AI/ML in heavily commented jupyter notebooks. In addition to the basics of ML and popular [deep learning](https://www.projectpro.io/article/deep-learning-architectures/996) techniques, there are also two sessions covered how to use the ML libraries [Keras](https://keras.io/) and [PyTorch](https://pytorch.org/), which offer the tools to run these models and other useful resources. Fuller detail on the bootcamp and a follow-on hackathon can be found in our EOS article [[Goldberger et al 2025](https://eos.org/science-updates/a-two-step-approach-to-training-earth-scientists-in-ai)].

Recently, the  bootcamp materials developed by PNNL scientists were updated for public release and made available on PNNL’s github [[IPID 33445, IR#373274](https://github.com/pnnl/AI_Modules_for_Earth_Science/tree/main)]. Each lesson folder contains a clearly ordered set of notebooks, a dedicated environment yaml file, datasets, and a PowerPoint which should be viewed first. A pre-lesson to the bootcamp not included in the github, introduction to python, was taught by Rob Hetland based on Texas A&M’s [python4geosciences](https://github.com/kthyng/python4geosciences) course covered syntax, data containers, basic logical control, reading and plotting data with Pandas and Xarray, georeferenced data and projections, and basic analysis techniques. 

## 1️⃣ Lesson 1: ML 101
*Lesson by TC Chakraborty & Sally Wang*

Lesson one covered the basics of machine learning such as the ML workflow and gave an overview of different machine learning methods. It dived deeper into clustering and tree-based algorithms. It gave examples of simple use-cases of supervised and unsupervised learning using SST data for predicting ENSO. 

{{< figure
  src="https://i.ibb.co/RpnyS0QZ/Picture1.png"
  alt="Surface maps of SST Anomaly"
  caption="Figure generated during bootcamp module ML101: Unsupervised Learning, applying K-Means composites for El Nino and La Nina cases. Based on tutorial created by Ben A. Toms for AMS 2020 Short Course."
>}}

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## 2️⃣ Lesson 2: Artificial Neural Networks
*Lesson by Maruti Mudunuru & TC Chakraborty*

Lesson two describes the concept of deep learning and covers shallow versus deep neural networks. It discusses the components of the neural network and the deep learning workflow. There were two activities for this lesson. Part one focused on the workflow aspect using a simple end-to-end data pipeline to pre-process, train, and predict hydrological model parameters from time-series data. The synthetic data and process model is based on the Soil & Water Assessment Tool (SWAT) hydrologic model. Part two returns to the ENSO dataset from lesson one. This time, the tutorial uses neural networks for unsupervised learning to try to detect ENSO phases from sea surface temperature data. It dives into clustering and autoencoder methods that rely on neural networks using monthly sea-surface temperature data to explore problems related to the El Nino Southern Oscillation (ENSO).

{{< figure
  src="https://i.ibb.co/3yrSj9yW/Picture2.png"
  alt="Two Plots related to model training, Epoch vs Loss and Ground Truth vs Prediction"
  caption="Figure generated during bootcamp module Lesson 2 Part 1: DNN Inverse Models (dnn_im_notebook.ipynb), input is river discharge and output is the SWAT model parameters. Once the loss function evens out for the validation dataset, cease running epochs to reduce risk of overfitting wave parameters. "
>}}

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## 3️⃣ Lesson 3: TensorFlow and Keras Library Packages
*Lesson by Erol Cromwell*

Lesson three provided an overview of the TensorFlow and keras library packages. TensorFlow is an end-to-end, open-source machine learning platform. For this lesson we focused on it’s python APIs. Keras is a user interface for deep learning, dealing with layers, models, optimizers, loss functions, metrics, and more. Keras makes TensorFlow simple and productive. 

The example in lesson three sets up and trains a deep neural network model (DNN A2 from [Cromwell et al. 2021](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2021.613011/full)) using Keras to estimate five subsurface (soil and geologic) permeability parameters from simulated watershed discharge time series data. This is an example of inverse modeling: using observed data (stream discharge) to infer causal factors (subsurface permeability). Subsurface permeability is one of the key parameters that determine the subsurface flow and transport processes in watershed models. However, this parameter is difficult and expensive to measure directly at the spatial extent and resolution required by fully distributed, physics-based watershed model. The linkages between permeability and stream flow provide a new opportunity to estimate subsurface permeability from stream flow monitoring data that are made available through monitoring networks. The data was generated using ensemble forward simulations of the Rock Creek watershed in Colorado using the Advanced Terrestrial Simulator (ATS).

{{< figure
  src="https://i.ibb.co/BVSqvBMW/Picture3.png"
  alt="Schematic of a DNN"
  caption="Schematic of DNN trained in lesson 3 using Keras. The input is the normalized discharge time-series data. The outputs from the DNN are five normalized permeability parameters. "
>}}

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## 4️⃣ Lesson 4: Convolutional Neural Networks
*Lesson by Maruti Mudunuru, Sam Dixon, Robin Cosby, & Andrew Geiss*

In lesson four, the keras pipeline is extended to include CNN along with hyperparameter tuning using KerasTuner for CNNs. Participants learned about the convolutional neural networks and its components, the difference between CNNs and DNNs, and also hyperparameter tuning to find optimal DNN and CNN architectures. The hands on exercise teaches how to develop a workflow to load, visualize, pre-process, and develop a CNN models revisiting the same synthetic data and SWAT model used in lesson 2.

{{< figure
  src="https://i.ibb.co/CpsfVwDB/Picture4.png"
  alt="Timeseries of streamflow"
  caption="Observed vs Modeled streamflow discharge using CNN. "
>}}

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## 5️⃣ Lesson 5: Generative Adversarial Networks
*Lesson by Andrew Geiss and Melissa Swift*

Lesson five defines generative models, their implementation, identifies practical concerns, describes combined loss functions, and discusses ethical concerns for generative ML. GAN are ML models that generate realistic data samples. The first example is the classic number generator, MNIST. 
The second demonstration is atmospheric science oriented and uses GAN to generate realistic samples of true color MODIS imagery. The training image samples are taken from a dataset of labeled marine cloud regimes and contain primarily low-level clouds over the subtropical oceans. GANS provide highly realistic but potentially false outputs, this leads to obvious ethical concerns in environmental science. This presentation deliberates ethical ML stewardship.

{{< figure
  src="https://i.ibb.co/hFVsxPgS/Picture5.png"
  alt="Nine tiles each with an image of a cloud from satellite generated by a GAN"
  caption="After 100,000 training updates the GAN can produce extremely convincing MODIS imagery. It generates a variety of cloud morphologies and even includes features like sunlight and shadow."
>}}

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## 6️⃣ Lesson 6: Pytorch Library Package
*Lesson by Sally Wang*

Lesson six covers the basics and key elements of pytorch in comparison with keras. In the hands on tutorial, annual mean meterology from MERRA2 is used to predict the annual surface PM2.5 over China during 2000-2017. The goal of this project is to see whether we can use annual meteorology (surface temperature, RH, precipitation, surface pressure, 10-m U wind and 10-m V wind) to predict/explain annual surface PM2.5 concentration. 

To challenge bootcamp participants, this exercise asks participants to try and add a 1-D convolutional layer (with kernel_size of 2, no padding, stride=1) before the fully connected layer to try to see whether the model performance improves after including a convolutional layer. If not, why?

{{< figure
  src="https://i.ibb.co/Z6pCzGSQ/Picture6.png"
  alt="Surface map of PM2.5 over China"
  caption="The surface PM2.5 and MERRA2 meteorology are regridded to a spatial resolution of 0.25º x 0.3125º."
>}}

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## 7️⃣ Lesson 7: Recurrent Neural Networks
*Lesson by Peishi Jiang*

Lesson 7 discusses sequence modeling, identifies the purpose of RNN, when to use long short-term memory models (LSTM), and the advantage of bidirectional LSTM. By learning information propagating back, biLSTMs effectively increase the amount of information available to the network. This lesson uses two examples to demonstrate the strength of sequence modeling: rainfall-runoff modeling with unidirectional LSTM and soil respiration modeling with bidirectional LSTM. Both of these examples use time series datasets and the pytorch library package covered in the previous lesson.

{{< figure
  src="https://i.ibb.co/whWd3Yj0/Picture7.png"
  alt="Plot of Lag Time versus Mutual Information"
  caption="Modeling soil respiration using ecosystem observations at AmeriFlux US-Hn1 site. "
>}}

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## What are the Next Steps

The success of the two-step approach speaks for itself. As of January 2026, three of the hackathon teams (made up of both participants and instructors from the bootcamp) published their projects. [Li et al 2024](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024GL110757) in GRL describes a data-driven framework for predicting aerosol-cloud interactions. Another team used a CNN model to identify open- versus closed-cell atmospheric convection in radar data, which helps explain distributions of clouds and rainfall, published in JGR Machine Learning and received a highlight in EOS [[Tian et al 2025](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024JH000486), [Folini 2025](https://eos.org/editor-highlights/machine-learning-provides-a-new-perspective-of-low-level-clouds)]. A third project also compared different classification models on classifying thermodynamic cloud phase in AMT [[Goldberger & Levin et al 2025](https://amt.copernicus.org/articles/18/5393/2025/)]. PNNL reported an uptick in proposals from its Earth scientists for various internal funding opportunities focused on leveraging AI/ML methods, signifying steps towards independent AI driven research.  

Whether organizing workforce training or learning ML/AI independently, success with AI like any skill takes time, but the effort is worthwhile, and a collaborative, cross-disciplinary environment accelerates such learning. Our two-step approach can rather be described as a pyramid, building a strong foundation leading to independence in stages: 
1.	**Community (Foundation)**
A strong community is essential. This includes having peers, mentors, and collaborators you can turn to for questions, feedback, and shared learning. This is finding your machine learning experts, data scientists, and domain scientists interested in the next frontier of AI/ML.
2.	**AI/ML Literacy**
The next level is developing a baseline understanding of AI/ML concepts—what methods exist, what they do well, and which approaches are most relevant to your research. This is the bootcamp.
3.	**Framing Research Questions**
With foundational knowledge in place, researchers can begin applying AI/ML by framing appropriate scientific questions, choosing appropriate methodologies, and curating data suited to those methods. This is the hackathon.
4.	**Independence (Apex)**
At the top of the pyramid is independence: the ability to analyze results critically, apply methods autonomously, and identify new scientific directions enabled by AI/ML. Independence will be identified through journal articles and proposals.

{{< figure
  src="https://i.ibb.co/Kc69YDpM/Picture9.png"
  alt="Pyramid describing steps for AI success"
  caption="Schematic of AI learning success. Generated by OpenAI with input from Lexie Goldberger"
>}}

