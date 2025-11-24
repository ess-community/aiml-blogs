---
title: "Foundation Models for Earth System - An Introduction"
date: 2025-11-24
tags: ["Foundational Model", "Earth",]
summary: "A beginner’s guide to the powerful models shaping the future of Earth science"
series: ["PaperMod"]
author: ["Phong Le"]
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: true
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
    # image: "https://www.amd.com/content/dam/amd/en/images/blogs/designs/projects/orbit-2-collaboration/orbit-2-figure-1.png"
    image: https://www.olcf.ornl.gov/wp-content/uploads/ORBIT2-OLCF.png
    caption: "Image source: [Oak Ridge National Laboratory - ORNL]"
    relative: false # when using page bundles set this to true
    hidden: false          # don't hide globally
    hiddenInList: true     # hide in list pages
    hiddenInSingle: false  # show inside post
social:
    fediverse_creator: "@Phong Le - ORNL"
editPost:
    URL: "https://github.com/ess-community/aiml-blogs/blob/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---
AI is revolutionizing the way we study and understand our planet. One of the most exciting developments in this space for years to come is *foundation models for Earth system*. In this post, we’ll break down what these models are and why they have so much potential.

### What is a foundation model?
> *A foundation model is an AI model trained on a massive amount of data (usually by unsupervised learning) and can be adapted to a wide range of tasks.*

They are a turning point in AI, replacing traditional models that were trained for a single purpose. Once pre-trained, foundation models can be adapted to numerous downstream applications with little to no additional training samples.

{{< figure
  src="../../images/DOFA1.png"
  alt="Generative"
  caption="A generative model learns features from the training data and can generate new, similar, and high-quality contents ([Source](https://x.com/iscienceluvr/status/1592860024657051649))."
>}}

### What are foundation models for Earth system?
> *They are foundation models designed for*

> Foundation models are powerful artificial intelligence (AI) models that are trained on a massive amount of data and can be adapted to a wide range of tasks.

There are 5 key characteristics of Foundation Models:

1. Pretrained (using large data and massive compute so that it is ready to be used without any additional training)
2. Generalized — one model for many tasks (unlike traditional AI which was specific for a task such as image recognition)
3. Adaptable (through prompting — the input to the model using say text)
4. Large (in terms of model size and data size e.g. GPT-3 has 175B parameters and was trained on about 500,000 million words, equivalent of over 10 lifetimes of humans reading nonstop!)
5. Self-supervised (see footnote 1) — no specific labels are provided and the model has to learn from the patterns in the data which is provided — see the cake illustration below.
