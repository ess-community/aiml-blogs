---
title: "Diffusion Models (Part 3): Denoising Diffusion Implicit Models"
description: "A step-by-step guide to DDIM"
summary: "A step-by-step guide to DDIM"
date: 2025-12-03
tags: ["Diffusion Model", "Generative", "Implicit Model", "Non-Markovian"]
author: "Phong Le"
series: ["Diffusion Models"]
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
    # image: "images/Earth_diffusion.jpg" # image path/url
    relative: false # when using page bundles set this to true
    hidden: false          # don't hide globally
    hiddenInList: true     # hide in list pages
    hiddenInSingle: false  # show inside post
editPost:
    URL: "https://github.com/ess-community/aiml-blogs/blob/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

[In Part 1]({{< relref "../Intro-Diffusion-Models-part1/index.md" >}}), we looked at how *Denoising Diffusion Probabilistic Models (DDPMs)*[^Ho2020] generate samples by adding random noise to data and then learning how to remove it. 
One of the most prevalent drawbacks of DDPMs is that they need a large number of iterations to generate a reasonably good looking image.

*Denoising Diffusion Implicit Models (DDIMs)*[^JSong2020] was invented to overcome this problem. They build on the same ideas behind DDPMs but take bigger, smarter steps, cutting down the time it takes to turn noise into a meaningful result.

In this article, we’ll break down how DDIMs work and why they speed things up. If you missed Part 1, you may want to check it out first for the basics.

Let’s get started!

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## From DDPMs to DDIMs

Recall that the original DDPM transition probability in the forward pass takes the form:
$$
q(\mathbf{x}\_t \vert \mathbf{x}\_{t-1}) = \mathcal{N}(\mathbf{x}\_t; \sqrt{1 - \beta\_t} \mathbf{x}\_{t-1}, \beta\_t\mathbf{I})
$$

where $\\{\beta\_t \in (0, 1)\\}\_{t=1}^T$ is the predefined variance schedule. 

Note that we can jump straight from the original sample $\mathbf{x}\_0$ to any noised version of the forward diffusion process $\mathbf{x}\_t$:
 
$$
\mathbf{x}\_t = \sqrt{\bar{\alpha}\_t}\mathbf{x}\_0 + \sqrt{1 - \bar{\alpha}\_t}\boldsymbol{\epsilon}_0
$$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}\_t = \prod\_{i=1}^t \alpha\_i$.

Here, the transition probability $q(\mathbf{x}\_t \vert \mathbf{x}\_{t-1})$ follows a Markov chain -- meaning that the probability of $\mathbf{x}\_t$ is only dependent on $\mathbf{x}\_{t-1}$, not any previous states.

The advantage of a Markovian structure is that the system is memoryless -- i.e., once we know $\mathbf{x}\_{t-1}$, we will know $\mathbf{x}\_{t-1}$. However, the downside is that a Markov chain can take many steps to converge.

DDIM overcomes this issue by using non-Markovian.

## Probability Distributions in DDIM


## Derivation of the Transition Distribution.

## Inference for DDIM


The reverse process in traditional diffusion models (like DDPM) involves hundreds or thousands of steps. The larger the number of timesteps, the more passes the image must go through the neural network, resulting in an increased computational load.

| **Feature** |	**DDPM** | **DDIM** |
|-------------|----------|----------|
| Reverse Process | Stochastic | Deterministic |
| Sampling Speed | Slow ($T \ge 1000$ steps)	| Fast ($T\approx 50$ steps) |
| Image Quality | Very High | Almost the same |
| Retraining Needed? | Yes, if you change steps | No, use same model |

**Quick Summary**:
- DDIM is a faster alternative to DDPM for image generation
- Uses a deterministic reverse process, reducing randomness
- Supports fewer steps without hurting quality much
- Works with existing pre-trained models

## References
[^Goodfellow:2014]: Goodfellow, I. et al., 2014. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661). *Advances in Neural Information Processing Systems (NeurIPS)*, 27, pp.2672–2680.
[^Kingma2014]: Kingma, D.P. & Welling, M., 2014. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). *Proceedings of the International Conference on Learning Representations (ICLR)* 2014.
[^Kingma2018]: Kingma, D.P. & Dhariwal, P., 2018. [Glow: Generative Flow with Invertible 1×1 Convolutions](https://arxiv.org/abs/1807.03039). *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*, Montréal, Canada.
[^Sohl-Dickstein2015]: Sohl-Dickstein, J. et al., 2015. [Deep unsupervised learning using nonequilibrium thermodynamics](https://arxiv.org/abs/1503.03585). *Proceedings of the 32$^{nd}$ International Conference on Machine Learning (ICML)*, PMLR, 37, pp.2256–2265.
[^Ho2020]: Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*, pp. 6840-6851.
[^lillog_diff]: https://lilianweng.github.io/posts/2021-07-11-diffusion-models
[^theaisummer]: https://theaisummer.com/diffusion-models
[^Lai2025]: Lai, C.-H. et al., 2025. [The Principles of Diffusion Models](https://www.arxiv.org/pdf/2510.21890).
[^Luo2022]: Luo, C., 2022. [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970).
[^Ozdemir]: Özdemir H., [Diffusion Models Explained with Math From Scratch](https://www.youtube.com/watch?v=fbJac4qQy04).
[^JSong2020]: Song, J., Meng, C., & Ermon, S., 2020. [Denoising diffusion implicit models](https://arxiv.org/abs/2010.02502). arXiv preprint arXiv:2010.02502.
[^Rombach2021]: R. Rombach, et al., 2021. [High-Resolution Image Synthesis with Latent Diffusion Models](https://www.computer.org/csdl/proceedings-article/cvpr/2022/694600k0674/1H1iFsO7Zuw), in 2022 IEEE/CVF Conference on CVPR, New Orleans, LA, USA, 2022.
[^YSong2020]: Song, Y., et al. 2020. [Score-based generative modeling through stochastic differential equations](https://arxiv.org/abs/2011.13456), arXiv preprint arXiv:2011.13456 (2020).
