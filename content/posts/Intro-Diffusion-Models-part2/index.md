---
title: "Diffusion Models: Principles and Applications in Earth Sciences - Part 2"
description: "Diffusion models are transforming how we analyze and predict complex Earth system processes"
summary: "Diffusion models for environmental science"
date: 2026-11-10
tags: ["Diffusion Model", "Weather Forecast", "Earth system"]
author: "Phong Le"
series: ["AI-ML"]
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: true
disableHLJS: true # to disable highlightjs
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
    # image: "images/Earth_diffusion.jpg" # image path/url
    # image: "https://cdn.satnow.com/community/AI_imagery_cover_638882532883884551.png"
    alt: "<alt text>" # alt text
    caption: "Source: satnow.com" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false          # don't hide globally
    hiddenInList: true     # hide in list pages
    hiddenInSingle: false  # show inside post
editPost:
    URL: "https://github.com/ess-aiml/blogs/blob/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

In [Part 1]({{< relref "../Intro-Diffusion-Models-part1/index.md" >}}), we explored the principles of diffusion models -- *how they take random noise and gradually transform it into meaningful data*. Here in Part 2, we look at how these models are being used to study our planet and why that could make a real difference for Earth science.

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## Probabilistic Weather Forecasting
*Weather influences nearly every part of our daily lives* — from personal safety to travel, work, and even what we choose to wear. But predicting it is far from simple.

The Earth's atmosphere is a *chaotic and constantly evolving system*, and we can’t observe everything happening within it. Even tiny measurement errors can amplify quickly, leading to very different forecast just a few days later.

Because of this inherent uncertainty, scientists do not aim to predict a single deterministic future. Instead, they use *ensemble forecasting*: numerical weather prediction (NWP) models are run many times with slightly different starting conditions to explore a range of plausible outcomes. This approach provides not just a forecast, but an *estimate of confidence* — how likely each scenario may be.

However, running these physics-based models at high resolution over and over requires massive computing power. As a result, detailed ensemble forecasts can be expensive and slow to generate.

This is where AI-based models come in. They can produce forecasts much faster — but until recently, they struggled to represent uncertainty in a physically meaningful way.

Now, that’s beginning to change...

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

### The next generation of AI weather models
GenCast[^Price2024] is a *probabilistic* weather forecasting system developed Google DeepMind. It use generative diffusion models to generate global 15-day ensemble forecasts at high spatial resolution (0.25$^\circ$).

We won’t go into performance metrics here -- those are well covered in the paper[^Price2024] and [Google DeepMind’s blog](https://deepmind.google/blog/gencast-predicts-weather-and-the-risks-of-extreme-conditions-with-sota-accuracy/).
Instead, we’ll focus on the foundation and principles behind GenCast: how diffusion models learn from historical weather data, and why this approach is well suited for representing uncertainty in Earth system forecasts.


<!--
## Precipitation Estimation from Satellites
{{< figure
  src="../../images/Guilloteau2024.gif"
  alt="Diffusion model"
  caption="Diffusion-based Ensemble Rainfall estimation from Satellite (DifERS). Source [Guilloteau et al., (2025)](https://ieeexplore.ieee.org/abstract/document/10912662)"
>}}

An example comes from [Guilloteau et al., (2025)](https://ieeexplore.ieee.org/abstract/document/10912662), who developed a generative diffusion framework called DifERS for producing ensembles of precipitation maps from multisensor satellite data. Their method combines physical insight with statistical learning to reconstruct detailed rainfall patterns from coarse satellite inputs. Two novelties of their method thus are: 1) the handling of the uncertainty through the generation of ensembles of equiprobable realizations and 2) the use of coincident measurements from different instruments and different platforms.

## Climate Downscaling
Because diffusion models explicitly model the distribution of states rather than just the mean response, they are particularly well-suited for capturing uncertainty, extremes, and multi-scale variability—features that are notoriously difficult for traditional deep learning architectures. For instance, recent studies [[Bassetti et al, (2024)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS004194), [Hess et al, (2025)](https://www.nature.com/articles/s42256-025-00980-5)] have shown that diffusion-based emulators can reconstruct fine-scale rainfall structures from coarse reanalysis data while preserving the physical coherence of storm systems, something most conventional downscaling models tend to blur.
-->
**To be continue...**

[^Price2024]: Price, I., Sanchez-Gonzalez, A., Alet, F. et al. [Probabilistic weather forecasting with machine learning](https://www.nature.com/articles/s41586-024-08252-9). *Nature* **637**, 84–90 (2025).
