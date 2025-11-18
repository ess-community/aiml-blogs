---
title: "Diffusion Models: Principles and Applications in Earth Sciences - Part 2"
description: "Diffusion models are transforming how we analyze and predict Earth system processes"
summary: "Diffusion models for environmental science"
date: 2025-11-10
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

 [In Part 1]({{< relref "../Intro-Diffusion-Models-part1/index.md" >}}), we explored the principles of *diffusion models* -- how they take random noise and gradually transform it into meaningful data. In this Part 2, we look at how diffusion models are being used to study our planet and why that could make a real difference.

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## 1. Probabilistic Weather Forecasting
Weather affects nearly every aspect of our daily lives -- from what we wear in the morning to how we plan for the days ahead. But predicting weather is far from simple, because the Earth's atmosphere is inherently *chaotic* and highly sensitive to *uncertainty*.

To address these issues, scientists rely on *ensemble forecasting*. Rather than predicting a single deterministic future, they run many simulations with slightly perturbed initial conditions. The resulting ensemble provides both a spectrum of plausible outcomes and a quantitative sense of forecast confidence.

However, *physics-based* numerical weather prediction (NWP) models come at a cost. Every additional member of the ensemble requires running a full simulation, making large ensembles both computationally expensive and slow to produce.

><mark> *Physics-based models use established laws of nature to simulate physical systems.*

This is where AI-driven forecasting comes in. They can generate forecasts much faster — but until recently, they struggled to represent uncertainty in physically meaningful ways.

That’s beginning to change...

### A Diffusion Model for Weather Forecast
In 2024, Google DeepMind introduced GenCast[^Price2024], a probabilistic weather forecasting system built on diffusion models. It provides:
- Global ensemble forecasts at 0.25$^\circ$ spatial resolution
- Forecast lead time up to 15 days ahead
- Improved performance for both weather and extreme events compared to ECMWF ENS

We won’t focus on performance here -- that is well covered in the paper[^Price2024] and [Google DeepMind’s blog announcement](https://deepmind.google/blog/gencast-predicts-weather-and-the-risks-of-extreme-conditions-with-sota-accuracy/).
Instead, we’ll focus on the principles of the GenCast.

### The Math
Let $\mathbf{X}^t$ denote the global weather state at time $t$.
GenCast assumes a second-order Markov approximation on the latent atmospheric dynamics, meaning that each future state depends only on the two most recent states:
\begin{equation}
P(\mathbf{X}^{t+1} | \mathbf{X}^{0:t}) \approx P(\mathbf{X}^{t+1} | \mathbf{X}^{t}, \mathbf{X}^{t-1})
\end{equation}
where $\mathbf{X}^{0:t} = (\mathbf{X}^{0}, \mathbf{X}^{1},\dots,\mathbf{X}^{t})$

The task of probabilistic weather forecasting from the present time $t=0$ into the future is to model the joint probability distribution $P(\mathbf{X}^{-1}, \mathbf{X}^{0:T} | \mathbf{O}^{\le 0})$, where $T$ is the forecast horizon and $\mathbf{O}^{\le 0}$ are observations made up to the forecast initialization time $t=0$.

Applying chain rule, this joint distribution can be written as:
<!--P(\mathbf{X}^{0:T} | {{\mathbf{O}}}^{\le 0}) = \underbrace{ P(\mathbf{X}^{0} | \mathbf{O}^{\le 0})}\_{\text{State inference}} \ \underbrace{P(\mathbf{X}^{1:T}| \mathbf{X}^{0})}\_{\text{Forecast model}}-->
<!--$$
P(\mathbf{X}^{-1}, \mathbf{X}^{0:T} | {{\mathbf{O}}}^{\le 0}) = \underbrace{ P(\mathbf{X}^{0}, \mathbf{X}^{-1} | \mathbf{O}^{\le 0})}\_{\text{state inference}} \ \underbrace{P(\mathbf{X}^{1:T}| \mathbf{X}^{0}, \mathbf{X}^{-1})}\_{\text{forecast model}}
$$-->
\begin{align}
P(\mathbf{X}^{-1}, \mathbf{X}^{0:T} | \mathbf{O}^{\le 0}) & = P(\mathbf{X}^{-1}, \mathbf{X}^{0}, \mathbf{X}^{1:T} | {{\mathbf{O}}}^{\le 0}) \\\
&= \underbrace{P(\mathbf{X}^{1:T} | \mathbf{X}^{0}, \mathbf{X}^{-1}, \mathbf{O}^{\le 0})}_\{{\color{red}\approx P(\mathbf{X}^{1:T} | \mathbf{X}^{0}, \mathbf{X}^{-1})}} \ P(\mathbf{X}^{0}, \mathbf{X}^{-1} | \mathbf{O}^{\le 0}) \\\
&\approx \underbrace{ P(\mathbf{X}^{0}, \mathbf{X}^{-1} | \mathbf{O}^{\le 0})}\_{\text{state inference}} \ \underbrace{P(\mathbf{X}^{1:T}| \mathbf{X}^{0}, \mathbf{X}^{-1})}\_{\text{forecast model}}
\end{align}

In practice, state inference $P(\mathbf{X}^{0}, \mathbf{X}^{-1} | \mathbf{O}^{\le 0})$ is handled simply by using ERA5 values derived from NWP systems.

The forecast model is factorized as:
\begin{equation}
p(\mathbf{X}^{1:T} \vert \mathbf{X}^0, \mathbf{X}^{-1}) = \prod_{t=0}^{T-1} p(\mathbf{X}^{t+1} \vert \mathbf{X}^t, \mathbf{X}^{t-1})
\end{equation}

In GenCast, each state $\mathbf{X}^t$ consists of 6 surface variables and 6 atmospheric variables at 13 pressure levels on a 0.25$^\circ$ lat-lon grid (*i.e.*, $\mathbf{X}^t \in \mathbb{R}^{(6+6\times 13)\times720\times1440}$).
Forecasts extend 15 days into the future, with 12 hour increments between steps, giving $T=30$.

{{< figure
  align=center
  src="../../images/GenCast_chain.jpg"
  alt="gencast model"
  caption="<small> **Forecast trajectory of GenCast up to 15 days**. If the current state is at Day-0, 18:00Z (UTC), the model uses the state from 12 hours earlier (Day-0 06:00Z) as context to predict the next state at Day-1 06:00Z. Each newly predicted state is then fed back into the model to generate subsequent time steps, progressing iteratively through Day-15 18:00Z.</small>"
>}}

So the critical piece of GenCast lies in how it models this conditional probability — turning uncertainty into a realistic distribution of future weather states.

### Diffusion models in GenCast
So far, we have focused on how GenCast formulates the forecasting problem as a conditional probability over future atmospheric states. The key question now becomes:

>How do we sample from this probability distribution in a way that remains physically realistic?

This is where diffusion models enter.

Diffusion models work by learning how to gradually transform random noise into a realistic atmospheric state — a reverse process called denoising. More formally, the model learns the score function:

{{< figure
  align=center
  src="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41586-024-08252-9/MediaObjects/41586_2024_8252_Fig1_HTML.png?as=webp"
  alt="Diffusion model"
  caption="Schematic of how GenCast produces a forecast based on diffusion model principles. Image source: [Price et al., (2025)](https://www.nature.com/articles/s41586-024-08252-9)"
  width="60%"
>}}
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

<!--| Type        | Variable name             | Short name |
| ----------- | ------------------------- | ---------- |
| Atmospheric | Geopotential              | $z$        |
| Atmospheric | Specific humidity         | $q$        |
| Atmospheric | Temperature               | $t$        |
| Atmospheric | U component of wind       | $u$        |
| Atmospheric | V component of wind       | $v$        |
| Atmospheric | Vertical velocity         | $w$        |
| Single      | 2 metre temperature       | 2$t$       |
| Single      | 10 metre u wind component | 10$u$      |
| Single      | 10 metre v wind component | 10$v$      |
| Single      | Mean sea level pressure   | $msl$      |
| Single      | Sea Surface Temperature   | $sst$      |
| Single      | Total precipitation       | $tp$       |-->
