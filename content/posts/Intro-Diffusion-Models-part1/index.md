---
title: "Diffusion Models (Part 1): Foundations and Principles"
description: "A beginner-friendly introduction to diffusion models: from noise to meaningful structure - with insights for Earth science"
summary: "A beginner’s guide to diffusion models"
date: 2025-10-28
tags: ["Diffusion Model", "Generative", "Earth system"]
author: "Phong Le"
series: ["AI-ML"]
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
    image: "images/Earth_diffusion.jpg" # image path/url
    relative: false # when using page bundles set this to true
    hidden: false          # don't hide globally
    hiddenInList: true     # hide in list pages
    hiddenInSingle: false  # show inside post
editPost:
    URL: "https://github.com/ess-community/aiml-blogs/blob/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

Diffusion models have become one of the most powerful tools in *Artificial Intelligence (AI)*. They’re the engines behind some of today's most advanced *generative systems* -- from creating realistic images, audio, text, and videos to designing new molecules and medicines, and even modeling complex climate and environmental systems.

There are already plenty of great articles that dive into the details of diffusion models -- and we’ll share some of our favorites along the way. In this series, we'll keep things accessible: *we focus on the <mark class="gray">core principles</mark> (in this post) and explore how diffusion models are being used in Earth and environmental sciences and why those applications are so promising (see [Part 2]({{< relref "../Intro-Diffusion-Models-part2/index.md" >}}), Part&nbsp;3, Part&nbsp;4).*

Let’s get started!

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## What are generative models?

>*<mark>Generative models are a type of AI system that learn the underlying structure of existing data and use it to create new content that resembles the original.</mark>*

What does this mean in practice? Suppose we have a dataset containing photos of dogs.
A generative model can study all those images to learn what makes a picture look like a dog -- *the shapes, colors, textures, and relationships between pixels*. Once trained, the model can then generate completely new, realistic images of dogs that did not exist in the original dataset.

Generative models are also *probabilistic*, *i.e.*, they don’t always produce the same output. Instead, they can create many different versions of an image or dataset, all slightly varied, but still realistic. This makes them especially useful for creative tasks, predictive simulations, and risk-based scientific modeling.

{{< quote-red >}}
<small>"Creating noise from data is easy; creating data from noise is generative modeling." </br>
-- **Song Y. et al., (2020)** </small>
{{< /quote-red >}}

{{< figure
  src="../../images/generative_modeling.jpg"
  alt="Generative"
  caption="A generative model learns features from the training data and can generate new, similar, and high-quality contents ([Source](https://x.com/iscienceluvr/status/1592860024657051649))."
>}}


There are different types of generative models, such as Generative Adversarial Networks[^Goodfellow:2014] (GANs), Variational Autoencoders[^Kingma2014] (VAEs), flow-based models[^Kingma2018], and diffusion models[^Sohl-Dickstein2015]<sup>,</sup>[^Ho2020]. Each type has its strengths and weaknesses, but diffusion models have recently shown outstanding performance in producing high-quality and realistic results. Their success largely comes from the ability to progressively refine noise, allowing diffusion models to capture complex data distributions and produce stable, high-fidelity results without the training instability common in other generative modeling approaches.

{{< quote-red >}}
**We focus on diffusion models in this series.**
{{< /quote-red >}}


{{< figure
  src="../../images/generative-overview.png"
  alt="Generative"
  caption="**Computation graphs of prominent generative models**. GANs use a generator $G(\mathbf{z})$ to produce data. VAEs use a probabilistic decoder $p_{\theta}(\mathbf{x}|\mathbf{z})$ to generate data. Flow-based models apply an invertible transformation $f^{-1}(\mathbf{z})$ to obtain data from latent variables. The transformations in these models are performed in a single forward pass through the neural network. Diffusion models instead gradually transform noise into data through a sequence of iterative denoising steps, reversing a learned diffusion process. <small>Image source: [Lil'Log](https://lilianweng.github.io/).</small>"
>}}

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## What are diffusion models?
Diffusion models are inspired by *non-equilibrium thermodynamics* -- specifically, how particles spread out or "*diffuse*" over time. The core idea behind them is simple: we gradually corrupt (*i.e.*, add noise to) clean data until it becomes completely random, then train a deep learning model to reverse this process and recover the original data.

><mark>*Diffusion models are a class of generative models that learn to reverse a gradual noising process applied to data, enabling them to generate realistic samples from the underlying data distributions by iteratively denoising random noise.*</mark>

{{< figure
  src="../../images/satellite_diffusion.gif"
  alt="Diffusion model"
>}}

In other words, diffusion models learn how to "*undo*" noise. Imagine taking a blurry or noisy satellite image and carefully sharpening it, one small step at a time, until continents and clouds slowly come back into focus. Each step removes a bit of noise, turning random patterns into something meaningful.

In principle, if we start from pure random noise, we should be able to keep applying the trained model until we obtain a sample that looks as if it were drawn from the training set.
That's it -- and yet this simple idea works incredibly well in practice.

{{< quote-blue >}}
*For a more intuitive explanation, check out [this article](https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models) -- it provides an interactive, step-by-step introduction that makes diffusion models much easier to grasp.*
{{< /quote-blue >}}


Diffusion models come in different forms, depending on whether the diffusion process is modeled in *discrete* or *continuous* time, and whether noise is removed through *probabilistic* or *deterministic* dynamics.

A breakthrough approach is the *Denoising Diffusion Probabilistic Model* (DDPM)[^Ho2020], which performs diffusion in discrete time. It models the generative process as a reverse <mark class="green">*Markov chain*</mark>, gradually denoising the sample through a fixed sequence of probabilistic transitions.

><mark class="green"> *A **Markov chain** is a discrete-time stochastic process where the next state depends only on the current state.*</mark>

Other diffusion formulations include DDPM-inspired variants such as *Denoising Diffusion Implicit Models* (DDIMs)[^JSong2020], which introduce a *non-Markovian* formulation that enables deterministic and faster sampling, and continuous-time *score-based* models[^YSong2020], which replace the discrete Markov chain with stochastic and ordinary differential equation perspectives to model the diffusion and denoising processes.
More recent approaches further optimize efficiency by performing diffusion in a compressed latent space (e.g., Latent Diffusion Models[^Rombach2021] - LBMs), or by unifying diffusion with flow-based or implicit *guidance techniques* for improved controllability and speed.

{{< quote-red >}}
**We focus on the DDPM in this post since it provides the most basic foundation.**
{{< /quote-red >}}


<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## How do DDPMs work?
Now, let’s explore how DDPMs actually work.
At their core, DDPMs involve two distinct stochastic processes in discrete time: a <mark class="blue">*forward diffusion pass*</mark> -- where noise is gradually added to data until it becomes purely random, and a <mark class="pink">*reverse denoising process*</mark> -- where the model learns to remove that noise step by step to reconstruct the original data.

{{< figure
  src="../../images/diffusion_processes.jpg"
  alt="Diffusion model"
>}}

### The Forward Process: Adding Noise
Suppose we have a real data sample $\mathbf{x}_0 \sim q(\mathbf{x})$. In the forward process, we gradually corrupt the data by adding small amounts of *Gaussian noise* over $T$ steps, producing a sequence of increasingly noisy samples $(\mathbf{x}_1, \dots, \mathbf{x}_T)$.
The amount of noise added at each step $t$ is controlled by a predefined *variance schedule* $\\{\beta\_t \in (0, 1)\\}\_{t=1}^T$.
$$
\begin{aligned}
q(\mathbf{x}\_t \vert \mathbf{x}\_{t-1}) &= \mathcal{N}(\mathbf{x}\_t; \sqrt{1 - \beta\_t} \mathbf{x}\_{t-1}, \beta\_t\mathbf{I}) \\\
q(\mathbf{x}\_{1:T} \vert \mathbf{x}\_0) &= \prod^T\_{t=1} q(\mathbf{x}\_t \vert \mathbf{x}\_{t-1})
\end{aligned}
$$

Here, $\mathcal{N}(\cdot,\cdot)$ denotes a [*normal distribution*](https://en.wikipedia.org/wiki/Normal_distribution).
As $t$ increases, the sample $\mathbf{x}_t$ becomes progressively noisier.
Eventually, when $T \rightarrow \infty$, $\mathbf{x}_T$ is indistinguishable from random noise.
Mathematically, we can write each step of this process as follows:
$$
\mathbf{x}\_t = \sqrt{1-\beta\_t}\mathbf{x}\_{t-1} + \sqrt{\beta\_t}\boldsymbol{\epsilon}\_{t-1} \quad \quad \text{where } \boldsymbol{\epsilon}\_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

><mark>*Note that when two components are independent, the variance of their sum is simply the sum of their variances.*</mark>

Since $\boldsymbol{\epsilon}\_{t-1}$ is standard Gaussian, if $\mathbf{x}\_{t-1}$ has zero mean and unit variance, then so does $\mathbf{x}_{t}$, because $\sqrt{1-\beta\_t}^2 + \sqrt{\beta\_t}^2=1$.

{{< figure
  src="../../images/forward_process.png"
  alt="Diffusion model"
  caption="Forward diffusion process."
>}}

In theory, if we normalize the original sample $\mathbf{x}\_{0}$ to have zero mean and unit variance, then the entire sequence $(\mathbf{x}_1, \dots, \mathbf{x}_T)$ will preserve these properties under the forward process. By the [*Central Limit Theorem*](https://en.wikipedia.org/wiki/Central_limit_theorem), $\mathbf{x}_T$ will approximate a standard Gaussian distribution as $T$ becomes sufficiently large.

><mark>*This scaling ensures that the variance remains stable throughout the diffusion process.*</mark>

In practice, inputs are typically scaled to a bounded range (*e.g.*, $[0,1]$ or $[-1,1]$), and this range must be known and consistent because the noise schedule is defined relative to the data's scale.

Another nice property of the above process is that we can jump straight from the original sample $\mathbf{x}_0$ to any noised version of the forward diffusion process $\mathbf{x}_t$ using a *reparameterization trick*</mark>.

Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}\_t = \prod\_{i=1}^t \alpha\_i$, then we can write the following:
$$
\begin{aligned}
\mathbf{x}\_t
&= \sqrt{\alpha\_t}\mathbf{x}\_{t-1} + \sqrt{1 - \alpha\_t}\boldsymbol{\epsilon}\_{t-1} \\\
&= \sqrt{\alpha\_t} {(\underbrace{\color{red}\sqrt{\alpha\_{t-1}}\mathbf{x}\_{t-2} + \sqrt{1 - \alpha\_{t-1}}\boldsymbol{\epsilon}\_{t-2}}_{\mathbf{x}\_{t-1}} \)} + \sqrt{1 - \alpha\_{t}}\boldsymbol{\epsilon}\_{t-1} \\\
&= {\color{red}\sqrt{\alpha\_t \alpha\_{t-1}} \mathbf{x}\_{t-2} + \sqrt{\alpha\_t (1-\alpha\_{t-1})}\boldsymbol{\epsilon}\_{t-2}} + \sqrt{1 - \alpha\_{t}}\boldsymbol{\epsilon}\_{t-1} \\\
&= \sqrt{\alpha\_t \alpha\_{t-1}} \mathbf{x}\_{t-2} + {\color{red}\sqrt{1 - \alpha\_t \alpha\_{t-1}} \bar{\boldsymbol{\epsilon}}\_{t-2} } \\\
&= \dots \\\
&= \sqrt{\alpha\_t \alpha\_{t-1}\dots\alpha\_1} \mathbf{x}\_{0} + \sqrt{1 - \alpha\_t \alpha\_{t-1}\dots\alpha\_1} \boldsymbol{\epsilon}_0 \\\
&= \sqrt{\bar{\alpha}\_t}\mathbf{x}\_0 + \sqrt{1 - \bar{\alpha}\_t}\boldsymbol{\epsilon}_0
\end{aligned}
$$

><mark>***Explanation in words:***</mark> We unroll the update rule step by step, combining the noise terms along the way, so that $\mathbf{x}\_t$ can be written directly in terms of $\mathbf{x}\_0$.
>Note that since $\boldsymbol{\epsilon}\_{t-2}, \boldsymbol{\epsilon}\_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, their weighted sum is also Gaussian with standard deviation $\sqrt{\alpha_t (1-\alpha_{t-1})+(1-\alpha_t)} = \sqrt{1-\alpha_t\alpha_{t-1}}$, and $\bar{\boldsymbol{\epsilon}}\_{t-2} \sim \mathcal{N}(\mathbf{0},\mathbf{I}).$

The forward diffusion process $q$ can therefore be written in closed form as:
$$
q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}\big(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I}\big)
$$

What this really tells us is that $\mathbf{x}_t$ never loses the original signal -- it’s just being covered by more and more Gaussian noise. The goal of diffusion models is figuring out how to peel that noise away again.

<center> <span style="letter-spacing: 0.5rem;">• • •</span> </center>

### The Reverse Process: Learning to Denoise
The reverse process works in the opposite direction -- *and this is where the magic happens*. Instead of adding noise, the reverse systematically removes it, step by step, gradually reconstructing the original data. Once trained, the model can start from pure Gaussian noise and iteratively apply this reverse procedure to generate new, realistic samples similar to $\mathbf{x}_0$.

In theory, the reverse diffusion process is defined as $q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t)$ -- meaning that given a noisy sample $\mathbf{x}\_t$, we would like to compute the distribution of the previous, slightly less noisy sample $\mathbf{x}\_{t-1}$. However, this distribution is *intractable* in practice because it depends on the entire (unknown) data distribution.

{{< figure
  src="../../images/reverse_process.png"
  alt="Diffusion model"
  caption="Reverse denoising process."
>}}

**Conditioning trick**

Another useful trick in diffusion models is that the reverse transition becomes *tractable* if we condition on the original data $\mathbf{x}\_{0}$. Since the forward process is fully known, we can apply *Bayes’ rule* to obtain a closed-form expression:
$$
\begin{aligned}
q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0) &= q(\mathbf{x}\_t \vert \mathbf{x}\_{t-1}, \mathbf{x}\_0) \frac{ q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_0) }{ q(\mathbf{x}\_t \vert \mathbf{x}\_0) } \\\
&= \mathcal{N}\big(\mathbf{x}\_{t-1}; {\tilde{\boldsymbol{\mu}}}\_t, {\tilde{\beta}\_t} \mathbf{I}\big)
\end{aligned}
$$
where $\tilde{\boldsymbol{\mu}}\_t = {\frac{1}{\sqrt{\alpha\_t}} \Big( \mathbf{x}\_t - \frac{1 - \alpha\_t}{\sqrt{1 - \bar{\alpha}\_t}} \boldsymbol{\epsilon}\_t \Big)}$ and $\tilde{\beta}\_t = {\frac{1 - \bar{\alpha}\_{t-1}}{1 - \bar{\alpha}\_t} \beta\_t}$.

This derivation relies on the *Markov property* of the forward process -- each state $\mathbf{x}\_t$ depends on the previous $\mathbf{x}\_{t-1}$, not on the original data $\mathbf{x}\_0$. Formally, $q(\mathbf{x}\_{t} \vert \mathbf{x}\_{t-1}, \mathbf{x}\_0) = q(\mathbf{x}\_{t} \vert \mathbf{x}\_{t-1})$.
Since all the factors in the Bayes' rule expression are Gaussian, multiplying them results in another Gaussian. Using $\mathcal{N}\big(x; \mu, \sigma\big) \propto \exp \left( \frac{-(x-\mu)^2}{2\sigma^2}\right),$ we can solve analytically for $\tilde{\boldsymbol{\mu}}\_t$ and $\tilde{\beta}\_t$ as shown above.

><mark class="gray">*If you’d like a quick walkthrough of this derivation, check out this [Lil'Log's post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models)[^lillog_diff]. For a full step-by-step version, see page 12 of this article[^Luo2022] and chapter 2 of this book[^Chan2024].*</mark>

What does this conditioning mean? It means that during training, since we know $\mathbf{x}\_0$, we can compute the exact noise that was added to get $\mathbf{x}\_t$. This allows us to create training pairs $(\mathbf{x}\_t, \mathbf{\epsilon})$, where $\mathbf{\epsilon}$ is the exact noise, and train a model to predict this noise.

**Why do we need deep learning?**

However, at generation time, we start from pure Gaussian noise and do not know $\mathbf{x}\_0$.
So we can no longer use the closed-form $q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0)$.

This is where deep learning comes into play.
We instead train a *neural network* $\mathbf{\epsilon}\_\theta(\mathbf{x}\_{t},t)$ to predict the noise added at each step. Once we have this noise estimate, we can recover an estimate of the clean signal and approximate the true reverse process:
$$
p\_\theta(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t) \approx q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t)
$$

><mark>*At each diffusion step, the neural network predicts the noise inside the current noisy sample and then subtracts it accordingly.*</mark>

Since each step in the forward diffusion adds only a small amount of Gaussian noise, the reverse steps can also be modeled as Gaussian transitions:

$$
p\_\theta(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t) = \mathcal{N} \big( \mathbf{x}\_{t-1}; \boldsymbol{\mu}\_\theta(\mathbf{x}\_t, t), \boldsymbol{\Sigma}\_\theta(\mathbf{x}\_t, t) \big)
$$

By applying this reverse transition from $t=T \rightarrow 0$, we gradually transform pure noise $\mathbf{x}\_T$ to a coherent, realistic sample that is similar to $\mathbf{x}\_0$:
$$
p\_\theta(\mathbf{x}\_{0:T}) = p\_\theta(\mathbf{x}\_T) \prod^T\_{t=1} p\_\theta(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t)
$$

Note that although the noise added during the forward diffusion is random, it is *not arbitrary* -- its structure comes from the underlying data. As a result, by learning to predict and remove this noise accurately, the model implicitly learns the structure of the original image $\mathbf{x}\_0$ and how to reconstruct it from noise.

<!--**In short:**
>- **Training:** we know $\mathbf{x}\_0$ → compute true noise → train a model to predict it
>- **Generation:** we start from pure Gaussian noise → the model predicts noise → remove noise step by step-->

<center> <span style="letter-spacing: 0.5rem;">• • •</span> </center>

### Train diffusion models
The goal of training a diffusion model is to make it assign *high probability* to real data. Formally, we want to maximize the *likelihood* of samples from the true data distribution:
$$
\max\_{\theta} \mathbb{E}\_{\mathbf{x}\_0 \sim q(\mathbf{x}\_0)} \Big[ \log p\_{\theta}(\mathbf{x}\_0) \Big]
$$
Here $q(\mathbf{x}\_0)$ is the real data distribution, and $p\_{\theta}(\mathbf{x}\_0)$ is the distribution modeled by the neural network.
However, the likelihood $p\_{\theta}(\mathbf{x}\_0)$ is intractable because the model generates data through a chain of latent noisy variables:
$$
p_{\theta}(\mathbf{x}\_0)
= \int p\_\theta(\mathbf{x}\_{0:T}) d\mathbf{x}\_{1:T}
$$

To solve this, diffusion models use a classical idea from variational inference -- the *Evidence Lower Bound (ELBO)*.

><mark class="pink">*We can’t compute the true likelihood, but we can compute a lower bound on it and train the model by maximizing that bound*.</mark>

ELBO is a computable lower bound on the true log-likelihood of data. We maximize it because doing so also maximizes the likelihood of real data — but in a way we can actually calculate.

$$
\begin{aligned}
\underbrace{\log p\_\theta(\mathbf{x}\_0)}\_{\text{Evidence}}
&= \log \int p\_\theta(\mathbf{x}\_{0:T}) d\mathbf{x}\_{1:T} = \log \int {\color{red} q(\mathbf{x}\_{1:T} \vert \mathbf{x}\_{0})} \frac{p\_\theta(\mathbf{x}\_{0:T})}{\color{red}{q(\mathbf{x}\_{1:T} \vert \mathbf{x}\_{0})}} d\mathbf{x}\_{1:T} \\\
&= \log \mathbb{E}\_{q(\mathbf{x}\_{1:T} \vert \mathbf{x}\_0)} \Bigg[\frac{p\_\theta(\mathbf{x}\_{0:T})}{q(\mathbf{x}\_{1:T} \vert \mathbf{x}\_{0})}\Bigg] \quad \quad \color{green}\small{\text{By definition: } \mathbb{E}\_{p(x)}[f(x)] = \int p(x)f(x)dx} \\\
&\ge \underbrace{\mathbb{E}\_{q(\mathbf{x}\_{1:T} \vert \mathbf{x}\_0)} \Bigg[ \log \frac{p\_\theta(\mathbf{x}\_{0:T})}{q(\mathbf{x}\_{1:T} \vert \mathbf{x}\_{0})} \Bigg]}\_{\text{Evidence Lower Bound} \ (ELBO\_{\theta})} \quad \quad \color{green}\small{\text{Apply Jensen's inequality (log is concave)}} \\\
&\rule{0pt}{2em} \color{blue}\small{\text{. . . We skip the details here for simplicity. At the end, we obtain:}} \\\
&\ge \underbrace{\mathbb{E}\_{q(\mathbf{x}\_{1} \vert \mathbf{x}\_0)} \Big[ \log p_{\theta}(\mathbf{x}\_{0} \vert \mathbf{x}\_{1})\Big] - D\_\text{KL}\big(q(\mathbf{x}\_{T}\vert\mathbf{x}\_0) \|| p\_\theta(\mathbf{x}\_{T}) \big) - \sum\_{t=2}^T \mathbb{E}\_{q(\mathbf{x}\_{t} \vert \mathbf{x}\_0)} \Big[ D\_\text{KL}\big(q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0) \parallel p\_\theta(\mathbf{x}\_{t-1} \vert\mathbf{x}\_t) \big)\Big]}\_{\text{Variational Lower Bound ($L\_{VLB}$)}}
\end{aligned}
$$

><mark class="gray">*For a complete derivation, check out this video[^Ozdemir] and this article[^Luo2022]*.</mark>

where $D\_\text{KL}(q||p)$ is the *Kullback–Leibler (KL) divergence*. Basically, it measures the similarity between two probability distributions. KL divergence is always positive and can be non-symmetric under the interchange of $p$ and $q$.

{{< figure
  src="../../images/ELBO.png"
  alt="Diffusion model"
  caption="Visualization of $\log p_{\theta}$ and $\text{ELBO}\_{\theta}$. The gap between the two curves is determined by the Kullback–Leibler divergence $D\_\text{KL}\big(q(\mathbf{x}\_{1:T} \vert \mathbf{x}\_{0}) \mid\mid p\_\theta(\mathbf{x}\_{1:T} \vert \mathbf{x}\_{0}) \big)$. <small>*Adapted from [Chan (2024)](https://arxiv.org/abs/2403.18103)*</small>."
  width=70%
>}}

To train the model, we instead minimize the negative log-likelihood bound:
$$
-\log p\_\theta(\mathbf{x}\_0) \le \underbrace{\mathbb{E}\_{q(\mathbf{x}\_1 \vert \mathbf{x}\_0)} \big[- \log p\_\theta(\mathbf{x}\_0 \vert \mathbf{x}\_1)\big]}\_{L\_0 \ (\text{reconstruction})} + \sum\_{t=2}^T \underbrace{\mathbb{E}\_{q(\mathbf{x}\_t \vert \mathbf{x}\_0)} \Big[ D\_\text{KL}\big(q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0) \parallel p\_\theta(\mathbf{x}\_{t-1} \vert\mathbf{x}\_t)\big)\Big]}\_{L\_{t-1} \ (\text{consistency})} + \underbrace{D\_\text{KL}\big(q(\mathbf{x}\_T \vert \mathbf{x}\_0) \parallel p\_\theta(\mathbf{x}\_T)\big)}\_{L\_T \ (\text{prior matching})}
$$

Here, $L_T$ is constant with respect to $\theta$ and can be ignored during training. The consistency term is a summation of many KL divergence terms. 
Every KL divergence term in $L_{LVB}$ (except for $L_0$) compares two Gaussian distributions and therefore they can be computed in closed form: 
$$
\begin{aligned}
D\_\text{KL}\big(q(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0) \parallel p\_\theta(\mathbf{x}\_{t-1} \vert\mathbf{x}\_t)\big) & = D\_\text{KL}\big(\mathcal{N}(\mathbf{x}\_{t-1}; \underbrace{\boldsymbol{\mu}\_t(\mathbf{x}\_t,\mathbf{x}\_0)}\_{\text{known}}, \underbrace{\sigma\_t^2 \mathbf{I}}\_{\text{known}}) \parallel \mathcal{N}(\mathbf{x}\_{t-1}; \underbrace{\boldsymbol{\mu}\_{\theta}(\mathbf{x}\_t)}\_{\text{neural net}}, \underbrace{\sigma\_t^2 \mathbf{I}}_\{\text{known}}) \big) \\\
&=\frac{1}{2\sigma_t^2} || \boldsymbol{\mu}\_t(\mathbf{x}\_t,\mathbf{x}\_0) - \boldsymbol{\mu}\_{\theta}(\mathbf{x}\_t)||^2
\end{aligned}
$$
The ELBO can also be simplified to absorb the reconstruction $L_0$ into the summation (see *Theorem 2.7* in this book[^Chan2024] for details).
This ultimately reduces to a simple and intuitive *loss*[^Ho2020]:
$$
\rm{ELBO}\_{\theta}(\mathbf{x}\_0,\boldsymbol{\epsilon}) = \mathbb{E}\_{\mathbf{x}\_0, \epsilon} \left[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha_t})} || \epsilon - \epsilon\_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}, t) ||^2 \right]
$$

To train the denoiser, we solve the optimization:
$$
\operatorname*{argmin}_{\theta} \ \mathbb{E}\_{\mathbf{x}\_0, \epsilon} \ \rm{ELBO}\_{\theta}(\mathbf{x}\_0,\boldsymbol{\epsilon})
$$

By minimizing this loss, the model learns to invert each step of the noising process. As training progresses, it becomes increasingly effective at removing noise from any noisy input $\mathbf{x}\_T$, enabling it to generate realistic samples starting from pure random noise.

The training and sampling algorithms in DDPM can be summarized as below:
<small>
<div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">

  <div style="border: 3px solid #00aeef; border-radius: 11px; padding: 0px; width: 49%;">
    <h4 style="margin: 0; text-align: center;">
      <span style="background: #00aeef; display:block; padding:4px;">
        Training DDPM
      </span>
    </h4>
    <ul style="margin-left: 0px; margin-top: 0px; list-style: none;">
      <li>1: <b>repeat</b>
      <li>2: &emsp;$\mathbf{x}_0 \sim q(\mathbf{x}_0)$
      <li>3: &emsp;$t\sim \text{Uniform}(\{1,...,T\})$
      <li>4: &emsp;$\boldsymbol{\epsilon} \sim \mathcal{N}(0,\mathbf{I})$
      <li>5: &emsp;take gradient descent step on: </br>
             &emsp;&emsp;&emsp; $\nabla_{\theta} \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon},t) \|^2$
      <li style="margin-bottom: -20px;">6: <b>until</b> converged
    </ul>
  </div>

  <div style="border: 3px solid lightsalmon; border-radius: 11px; padding: 0px; width: 49%;">
    <h4 style="margin: 0; text-align: center;">
    <span style="background: lightsalmon; display:block; padding:4px;">
        Sampling DDPM
    </span>
    </h4>
    <ul style="margin-left: 0px; margin-top: 0px; list-style: none;">
      <li>1: $\mathbf{x}_T \sim \mathcal{N}(0,\mathbf{I})$
      <li>2: <b>for $t=T,...,1$ do</b>
      <li>3: &emsp; $\mathbf{z} \sim \mathcal{N}(0,\mathbf{I})$ if $t>1$, else $\mathbf{z}=0$
      <li>4: &emsp; $\mathbf{x}_{t-1} = \tfrac{1}{\sqrt{\alpha_t}}\bigl(\mathbf{x}_t - \tfrac{1 - \alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(\mathbf{x}_t, t)\bigr) + \sigma_t \mathbf{z}$
      <li>5: <b>end for</b>
      <li style="margin-bottom: -20px;">6: <b>return</b> $\mathbf{x}_0$
    </ul>
  </div>
</div>
</small>

><mark class="gray">*If you’d like to explore the complete mathematical derivation, check out these excellent resources[^lillog_diff]<sup>,</sup>[^Luo2022]<sup>,</sup>[^Chan2024]<sup>,</sup>[^Ozdemir]<sup>,</sup>[^theaisummer]<sup>,</sup>[^Lai2025]. Each provides a detailed explanation of the theory and intuition behind diffusion models.*</mark>

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## Summary
- We gradually add noise to data (the *forward process*).
- The model learns to remove the noise (the *reverse process*).
- Training aims to *maximize the likelihood* of real data (or equivalently, *minimize the negative log-likelihood*).
- The exact likelihood is intractable, we instead minimize a *lower bound*.

</br>

{{< quote-blue >}}
In [Part 2]({{< relref "../Intro-Diffusion-Models-part2/index.md" >}}), we'll dive into how diffusion models are applied in Earth 🌎 sciences.
{{< /quote-blue >}}

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
[^Chan2024]: Chan, S., 2024. [Tutorial on Diffusion Models for Imaging and Vision](https://dl.acm.org/doi/abs/10.1561/0600000112). *Found. Trends. Comput. Graph. Vis.* **16**, 4, 322–471.
