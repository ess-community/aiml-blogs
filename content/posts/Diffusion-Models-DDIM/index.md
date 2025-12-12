---
title: "Diffusion Models (Part 3): Denoising Diffusion Implicit Models"
description: "An introduction to DDIMs - a faster and more efficient alternative to DDPMs."
summary: "An introduction to DDIMs - a faster and more efficient alternative to DDPMs."
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
    image: "images/Diifusion_Models_AI_modern.webp" # image path/url
    caption: "[Image source](https://www.dataexpertise.in/diffusion-models-ultimate-guide-generative-ai/)"
    relative: false # when using page bundles set this to true
    hidden: false          # don't hide globally
    hiddenInList: true     # hide in list pages
    hiddenInSingle: false  # show inside post
editPost:
    URL: "https://github.com/ess-community/aiml-blogs/blob/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

[In Part 1]({{< relref "../Intro-Diffusion-Models-part1/index.md" >}}), we explored *Denoising Diffusion Probabilistic Models (DDPMs)*[^Ho2020] and saw how they can turn pure noise into meaningful data. 
The idea is beautiful, but there’s a catch: ***sampling is slow***. 
A DDPM often needs hundreds or even thousands of small denoising steps to create one sample.

*Denoising Diffusion Implicit Models (DDIMs)*[^JSong2020] were introduced to fix this issue. They use the same basic diffusion idea as DDPMs, but change the way we sample so that we can generate good samples in far fewer steps.
<!--Even better, they work with the same trained DDPM model, so no retraining is required.-->

In this post, we’ll look at how DDIMs work and why they’re so much faster.
We’ll touch on some math, but only enough to develop an intuitive understanding of what’s going on.

{{< quote-red >}}
*If you’re new to diffusion models, we strongly recommend reading [Part 1]({{< relref "../Intro-Diffusion-Models-part1/index.md" >}}) first, as we will reuse some concepts from there.*
{{< /quote-red >}}

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## Notations
| **Symbols** |	**Meaning** |
|-------------|-------------|
| $T \in \mathbb{N}_{>0}$ | Number of diffusion steps to reach pure noise |
| $\mathbf{x}_0$ | Original clean sample	|
| $\mathbf{x}_t$ | Noisy sample at step $t$	|
| $\mathbf{x}\_{1:T}$ | Sequence of the samples $(\mathbf{x}_1, \dots, \mathbf{x}_T)$	|
| $\\{\beta\_t \in (0, 1)\\}\_{t=1}^T \quad \quad$ | Variance (noise) schedule	|
| $\mathcal{N}(\boldsymbol{\mu}, \sigma^2)$ | Normal (Gaussian) distribution with mean $\boldsymbol{\mu}$ and variance $\sigma^2$ |
| $q(\mathbf{x}_i \mid \mathbf{x}_j)$ | Transition distribution from state $\mathbf{x}_j$ to state $\mathbf{x}_i$ |


## The Problem with DDPMs
To see why DDIMs are helpful, let's do a short recap of DDPMs and their main limitation.

Suppose we have a real data sample $\mathbf{x}_0 \sim q(\mathbf{x})$.
In the forward process of a DDPM, we gradually corrupt this data by adding small amounts of Gaussian noise over $T$ steps, producing a sequence of increasingly noisy samples $(\mathbf{x}_1, \dots, \mathbf{x}_T)$.

Recall the [DDPM forward transition]({{< relref "../intro-diffusion-models-part1/index.md#the-forward-process-adding-noise" >}}) between steps:
\begin{equation}
q(\mathbf{x}\_t \vert \mathbf{x}\_{t-1}) = \mathcal{N}(\mathbf{x}\_t; \sqrt{1 - \beta\_t} \mathbf{x}\_{t-1}, \beta\_t\mathbf{I})
\label{eq:ddpm-forward}
\end{equation}

This equation describes how the model transitions from step $t-1$ to $t$:
- It keeps most of the previous sample, scaling $\mathbf{x}\_{t-1}$ by $\sqrt{1 - \beta\_t}$;
- It adds a small amount of fresh Gaussian noise with variance $\beta_t$;

Mathematically, each forward step can be written as:
\begin{equation}
\mathbf{x}\_t = \sqrt{1 - \beta\_t}\mathbf{x}\_{t-1} + \sqrt{\beta\_t}\boldsymbol{\epsilon}\_{t-1} \quad \quad \text{where } \boldsymbol{\epsilon}\_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
\end{equation}

Over steps, this forms a [Markov chain]({{< relref "../intro-diffusion-models-part1/index.md#markov-chain" >}}): *each state $\mathbf{x}\_{t}$ depends only on the immediately preceding state $\mathbf{x}\_{t-1}$, not on earlier steps*.

We can also directly relate a noisy sample $\mathbf{x}\_{t}$ back to the original clean sample $\mathbf{x}\_{0}$ (see the [reparameterization trick]({{< relref "../intro-diffusion-models-part1/index.md#reparameterization-trick" >}})):
\begin{equation}
q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}\big(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I}\big)
\label{eq:reparameter}
\end{equation}
where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}\_t = \prod\_{i=1}^t \alpha\_i$.

A major benefit of a Markov chain is its simplicity: *the next state can be sampled using only the current one*.

However, there is a flaw.
Because $\beta_t$ must be small to ensure stability, DDPMs require many $(T \gg 0)$ tiny increments to reach the fully noised state $\mathbf{x}\_T \sim \mathcal{N}(0,\boldsymbol{I})$, and the reverse process needs an equal number of steps (i.e., $T$) to traverse the entire chain for denoising.
<mark class="pink">This makes generating samples in DDPMs very slow.</mark>


<div style="border: 3px solid seagreen; border-radius: 10px; padding: 0px; width: 99%;">
<b style="margin: 0; text-align: center;">
<span style="background: seagreen; display:block; padding:4px; border-radius: 0px;">
    Example - Walking in a forest    
</span>
</b>
<div style="margin-top: 6px;"> </div>

<div style="padding: 0px 12px 8px 12px;text-align: justify;font-style: italic;">

Imagine you are walking through a dense forest. The trees block your view, so you can only see the ground right in front of you. If each step is based on your current position, you can move only in small, cautious increments because you have no sense of where the trail leads. <b>This is like a Markov chain (memoryless)</b>.

<div style="margin-top: 12px;"> </div>

Now imagine you also carry a map showing where you started and where the trail leads. With this extra “memory”, you can take larger, more confident steps -- even skip intermediate ones -- because you know the general direction. <b>This is like a non-Markovian process (with memory)</b>.
</div>
<div style="margin-top: -24px;"> </div>
</div>

<br>
<!--Because $\beta\_t$ is small at each timestep, DDPMs require many tiny increments to reach the fully noised state $\mathbf{x}\_T \sim \mathcal{N}(0,\boldsymbol{I}).$-->

DDIMs address the sampling inefficiency of DDPMs by formulating a <mark class="orange">*non-Markovian*</mark> process, enabling deterministic and flexible sampling schedules.

><mark class="orange">*A non-Markovian process is a system whose future behavior depends not only on its current state but also on its past history -- meaning the process has memory.*</mark>

{{< quote-blue >}}
**Why memory matters?** <br>
Memory allows a process to retain information from its past, which often influences its future evolution. Non-Markovian structures capture these dependencies, enabling more accurate modeling of system dynamics, smoother trajectories, and better predictions compared to memoryless (Markovian) systems.
{{< /quote-blue >}}

Let’s explore how DDIMs do this!

## Non-Markovian Forward Process
To generalize the DDPM forward process, the DDIM paper[^JSong2020] first changes the notation slightly. Instead of using the coefficient $\alpha_t$ directly, we express each transition in terms of the ratio $\color{red}\frac{\alpha_t}{\alpha_{t-1}}$. Thus, Eqn \eqref{eq:ddpm-forward} becomes:
\begin{equation}
q(\mathbf{x}\_t \vert \mathbf{x}\_{t-1}) = \mathcal{N}\left(\mathbf{x}\_t; \sqrt{{\color{red}\frac{\alpha_t}{\alpha_{t-1}}}} \mathbf{x}\_{t-1},  \Big(1-{\color{red}\frac{\alpha_t}{\alpha_{t-1}}}\Big)\mathbf{I}\right)
\end{equation}

This change does not have any physical meaning but simplifies notation. Under this form, the product term in Eqn \eqref{eq:reparameter} is simplified to:

$$
\bar{\alpha_t}=\prod\_{i=1}^t {\color{red}\frac{\alpha\_i}{\alpha\_{i-1}}}={\alpha_t} \quad \quad ; \text{assuming} \ \alpha_0=1
$$

Here, each intermediate $\alpha_i$ cancels, leaving only $\alpha_t$.
Using the [reparameterization trick]({{< relref "../intro-diffusion-models-part1/index.md#reparameterization-trick" >}}), the forward process becomes:
\begin{equation}
q(\mathbf{x}_t \vert \mathbf{x}\_0) = \mathcal{N}\big(\mathbf{x}\_t; \sqrt{{\color{red}\alpha_t}} \mathbf{x}_0, (1 - {\color{red}\alpha_t})\mathbf{I}\big)
\end{equation}

>*<mark>**Note:** The noisy marginals are the same as in the original DDPM. That means any diffusion model trained with the DDPM objective can be reused in the DDIM framework without retraining. The difference will come from how we define the reverse process.</mark>*

### Deterministic Reconstruction
The key idea behind DDIM is to make the reverse process less random (or even fully deterministic), while still matching the same noisy distributions $q(\mathbf{x}_t \vert \mathbf{x}\_0)$.

From the marginal we just saw, we can write:
\begin{equation}
\mathbf{x}\_t = \sqrt{\alpha\_t}\mathbf{x}\_0 + \sqrt{1 - \alpha\_t}\boldsymbol{\epsilon} \quad \quad ; \text{where} \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0,\boldsymbol{I})
\label{eq:6}
\end{equation}

Similarly, for the previous step:
\begin{equation}
\mathbf{x}\_{t-1} = \sqrt{\alpha\_{t-1}}\mathbf{x}\_0 + \sqrt{1 - \alpha\_{t-1}}\boldsymbol{\epsilon}
\label{eq:7}
\end{equation}

<!--An important trick used here is to replace $\boldsymbol{\epsilon}$ by something so that $\mathbf{x}\_{t-1}$ is no longer $\mathbf{x}\_0$ perturbed by white noise.-->
<mark class="blue">The trick is that we **reuse the same noise** $\boldsymbol{\epsilon}$ at both steps.</mark> Solving Eqn \eqref{eq:6} for $\boldsymbol{\epsilon}$ gives:

$$
\boldsymbol{\epsilon} = \frac{\mathbf{x}\_t - \sqrt{\alpha\_t} \mathbf{x}\_0}{\sqrt{1 - \alpha\_{t}}}
$$

Substituting this into Eqn \eqref{eq:7}, we obtain:
\begin{equation}
\mathbf{x}\_{t-1} = \sqrt{\alpha\_{t-1}}\mathbf{x}\_0 + \sqrt{1 - \alpha\_{t-1}} \color{red}\left(\frac{\mathbf{x}\_t - \sqrt{\alpha\_t} \mathbf{x}\_0}{\sqrt{1 - \alpha\_{t}}} \right)
\label{eq:8}
\end{equation}

This formulation removes the need for explicit sampling from a Gaussian at every timestep as in DDPMs. Instead, $\mathbf{x}_{t-1}$ is now a deterministic function of $\mathbf{x}_t$ and an estimate of $\mathbf{x}_0$.
This property forms the foundation for more efficient sampling.

<div style="border: 3px solid lightsalmon; border-radius: 10px; padding: 0px; width: 99%;">
<h4 style="margin: 0; text-align: center;">
<span style="background: lightsalmon; display:block; padding:4px; border-radius: 0px;">
    Why does this make sampling faster?
</span>
</h4>
<div style="margin-top: 6px;"> </div>

<div style="padding: 0px 12px 8px 12px;text-align: justify;font-style: italic;">
Because the update is now a deterministic transformation, the sampler no longer relies on small stochastic corrections. This makes long-range jumps through the diffusion trajectory feasible.
</div>
<div style="margin-top: 0px;"> </div>
</div>

<br>

### A family of inference processes
The DDIM paper defines not just one process, but a family of possible inference processes. Each member of this family is controlled by a vector $\sigma \in \mathbb{R}^T_{\ge 0}$, which decides how much extra noise we allow at each step:

\begin{equation}
q_{\sigma}(\mathbf{x}\_{1:T} \vert \mathbf{x}\_0) = q_{\sigma}(\mathbf{x}\_T \vert \mathbf{x}\_0) \prod^T\_{t=2} q_{\sigma}(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0)
\end{equation}
where the terminal distribution is defined as
$$
q_{\sigma}(\mathbf{x}\_T \vert \mathbf{x}\_0) = \mathcal{N}\big(\sqrt{\alpha\_T} \mathbf{x}_0, (1 - \alpha\_T)\mathbf{I}\big)
$$

The forward process is also Gaussian and can be derived from Bayes’ rule:

\begin{equation}
q_{\sigma}(\mathbf{x}\_{t} \vert \mathbf{x}\_{t-1}, \mathbf{x}\_0) = \frac{q_{\sigma}(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0) q_{\sigma}(\mathbf{x}\_{t} \vert \mathbf{x}\_0)}{q_{\sigma}(\mathbf{x}\_{t-1} \vert \mathbf{x}\_0)}
\label{eq:ddim-forward}
\end{equation}

You don’t need to memorize these formulas. The main takeaway is:
- We now allow transitions that depend on both the current state $\mathbf{x}\_t$ and the original data $\mathbf{x}\_0$.
- This makes the process non-Markovian (it has "memory").

{{< figure
  src="../../images/ddim_inferences.png"
  alt="Diffusion model"
  caption="Graphical models for diffusion (top) and non-Markovian (bottom) inference models. <small>*Adapted from [Song et al (2020)](https://arxiv.org/abs/2010.02502)*</small>."
>}}

## Generative Processes
The most important requirement in DDIM is: 
><mark class="blue">For every time step $t$, the marginal distribution $q_{\sigma}(\mathbf{x}\_{t} \vert \mathbf{x}\_0)$ should have the same form as in a DDPM:</mark> 

\begin{equation}
q_{\sigma}(\mathbf{x}_{t} \vert \mathbf{x}\_0) = \mathcal{N}\big(\sqrt{\alpha\_{t}} \mathbf{x}_0, (1 - \alpha\_{t})\mathbf{I}\big)
\label{eq:9}
\end{equation}

<!--The reason is that ultimately we want $q_{\sigma}(\mathbf{x}\_t \vert \mathbf{x}\_0)$ to become pure white noise when $t=T$ and the original sample when $t=0$.-->
Why do we want this?
- At $t=0$, we want $\mathbf{x}\_t$ to reduce to the original data $\mathbf{x}\_0$.
- At $t=T$, we want $\mathbf{x}\_T$ to look like almost pure Gaussian noise.
- Keeping the same form makes it possible to reuse DDPM-trained models.


There are many different choices of the reverse transition distribution $q\_{\sigma}(\mathbf{x}_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0)$, but only some of them can ensure that $q\_{\sigma}(\mathbf{x}\_t \vert \mathbf{x}\_0)$ has the desired structure we want above in Eqn \eqref{eq:9}.
For this purpose, the reverse transition distribution in DDIM is chosen as follows[^JSong2020]:

\begin{equation}
q_{\sigma}(\mathbf{x}_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0) = \mathcal{N}\big(\sqrt{\alpha\_{t-1}}\mathbf{x}\_0 + \sqrt{1 - \alpha\_{t-1}-\sigma_t^2} \left(\frac{\mathbf{x}\_t - \sqrt{\alpha\_t} \mathbf{x}\_0}{\sqrt{1 - \alpha\_{t}}} \right), \sigma_t^2\mathbf{I}\big)
\end{equation}

><mark>*Check out this tutorial[^Chan2024] to see the detailed derivation of this transition distribution.*</mark>

Here, the magnitude of $\sigma_t$ controls how much fresh noise is injected at each step. 
If $\sigma=0$, the process becomes fully deterministic, <mark class="orange">**and we get a DDIM**</mark>.
If $\sigma$ is chosen to match the DDPM posterior variance, <mark class="blue">**we recover DDPM sampling**</mark>.

### Inference for DDIM
Now let’s see how DDIM actually uses a neural network to go from noise back to data.

From the marginal:
\begin{equation}
\underbrace{\mathbf{x}\_t}\_{\text{given}} = \underbrace{\sqrt{\alpha\_t}\mathbf{x}\_0}\_{\text{want to find}} + \underbrace{\sqrt{1 - \alpha\_t}\boldsymbol{\epsilon}}\_{\text{estimated by network}}
\end{equation}

We can rearrange to solve for the clean sample $\mathbf{x}\_0$:
\begin{equation}
\mathbf{x}\_0 = \frac{1}{\sqrt{\alpha\_t}} \left( \mathbf{x}\_t - \sqrt{1 - \alpha\_t}\boldsymbol{\epsilon} \right) \\\
\end{equation}

<!--This can be approximated by a neural network:-->
In practice, we don’t know $\boldsymbol{\epsilon}$, so we train a neural network to predict the noise. Plugging this estimate in, we get a prediction of the clean sample:

\begin{equation}
{\color{red}f\_{\theta}^{(t)}(\mathbf{x}\_t)} = \frac{1}{\sqrt{\alpha\_t}} \left( \mathbf{x}\_t - \sqrt{1 - \alpha\_t} \cdot {\color{red}\boldsymbol{\epsilon}\_{\theta}^{(t)}(\mathbf{x}\_t)} \right)
\end{equation}

There are two new terms in this equation. The first one is $\color{red}\boldsymbol{\epsilon}\_{\theta}^{(t)}(\mathbf{x}\_t)$, which replaces $\boldsymbol{\epsilon}$. It is the estimate of the noise based on the current input $\mathbf{x}\_t$.
The second term is the denoised estimator $\color{red}f\_{\theta}^{(t)}(\mathbf{x}\_t)$, which is a prediction of the true signal $\mathbf{x}\_0$ given $\mathbf{x}\_t$.

Returning to the transition distribution $q\_{\sigma}(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0)$, if we do not have access to $\mathbf{x}\_0$, we can replace it with $\color{red}f\_{\theta}^{(t)}(\mathbf{x}\_t)$:

\begin{equation}
\begin{aligned}
p\_{\theta}(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t) &= q\_{\sigma}(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, {\color{red}f\_{\theta}^{(t)}(\mathbf{x}\_t)}) \\\
&= \mathcal{N}\big(\sqrt{\alpha\_{t-1}} \left( \frac{\mathbf{x}\_t - \sqrt{1 - \alpha\_t}\color{red}\boldsymbol{\epsilon}\_{\theta}^{(t)}(\mathbf{x}\_t)}{\sqrt{\alpha\_t}} \right) + \sqrt{1 - \alpha\_{t-1}-\sigma_t^2} \cdot {\color{red}\boldsymbol{\epsilon}\_{\theta}^{(t)}(\mathbf{x}\_t)}, \sigma_t^2\mathbf{I}\big)
\end{aligned}
\label{eq:reverse-eq}
\end{equation}

<mark>**The process can be summarized as follows**:</mark>
- Take a noisy sample $\mathbf{x}\_t$;
- Use the network to predict the noise $\boldsymbol{\epsilon}\_{\theta}^{(t)}(\mathbf{x}\_t)$;
- Use that to estimate the clean sample $\mathbf{x}\_0$;
- Use the DDIM update rule to compute the next sample $\mathbf{x}\_{t-1}$.

For the special case where $t=1$, we define:
$$
p\_{\theta}(\mathbf{x}\_{0} \vert \mathbf{x}\_1) = \mathcal{N}\big( {\color{red}f\_{\theta}^{(1)}(\mathbf{x}\_1)}, \sigma_1^2\mathbf{I} \big)
$$
meaning that we add some Gaussian noise (with covariance $\sigma_1^2\mathbf{I}$) so that the generative process is supported everywhere.

Even though the reverse process has changed, the resulting training objective turns out to be equivalent to the DDPM objective up to a constant. So the same trained model can be used for both DDPM and DDIM sampling.

\begin{equation}
J\_{\sigma}(\boldsymbol{\epsilon}\_{\theta}) = \mathbb{E}\_{\mathbf{x}\_{0:T} \sim q(\mathbf{x}\_{0:T})} \left[ \log q_\sigma(\mathbf{x}\_{1:T} \vert \mathbf{x}_0 ) - \log p\_{\theta}(\mathbf{x}\_{0:T}) \right]
\end{equation}

### Accelerated sampling
In a traditional DDPM, a full generative trajectory requires iterating through all $T$ diffusion steps. However, DDIM introduces a key insight: *the sampling schedule (which timesteps we visit during generation) is not tied to the original forward-noising schedule*. This flexibility allows us to define a subset of timesteps $\tau = \\{\tau_1,\dots,\tau_S\\}$ and run the reverse process only at those points.

{{< figure
  src="../../images/ddim_accelerated.png"
  alt="Diffusion model"
  caption="Graphical model for accelerated generation, where $\tau=[1, 3]$. <small>*Image source: [Song et al (2020)](https://arxiv.org/abs/2010.02502)*</small>."
  width=80%
>}}

In other words, instead of stepping through all $T$ noise levels, we can "jump" between them while still following a coherent generative path. Each DDIM update remains stable and predictable because it is derived from the deterministic structure of the transition.
When the length of this sampling trajectory $S \ll T$, we can achieve a significant increase in computational efficiency while still preserving high sample quality.

This is the foundation of accelerated sampling in DDIM and one of the main reasons it is widely used in practice.


## DDPM vs DDIM: Key Differences
The DDPM reverse update (in one common form) looks like:
\begin{equation}
\mathbf{x}\_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}\_t}} \boldsymbol{\epsilon}\_{\theta}^{(t)}(\mathbf{x}\_t) \right) + \\sqrt{\tilde{\beta}_t} \boldsymbol{\epsilon}\_t
\end{equation}

This update is stochastic -- even with the same starting noise, repeated runs give different trajectories.

From the DDIM derivation, the reverse update can be written as:
\begin{equation}
\mathbf{x}\_{t-1} = \sqrt{\alpha\_{t-1}} \underbrace{\left(\frac{\mathbf{x}\_t - \sqrt{1 - \alpha\_t} {\color{red}\boldsymbol{\epsilon}\_{\theta}^{(t)}(\mathbf{x}\_t)} }{\sqrt{\alpha\_t}}\right)}\_{\text{predicted} \ x\_0} + \underbrace{\sqrt{1 - \alpha\_{t-1} - \sigma\_t^2} \cdot {\color{red}\boldsymbol{\epsilon}\_{\theta}^{(t)}(\mathbf{x}\_t)}}\_{\text{direction pointing to } \mathbf{x}\_t} + \underbrace{\sigma\_t \boldsymbol{\epsilon}\_t}\_{\text{random noise}} 
\label{eq:sample-eq-gen}
\end{equation}

While both DDPM and DDIM use $\mathbf{x}\_t$ and $\boldsymbol{\epsilon}\_{\theta}^{(t)}(\mathbf{x}\_t)$ in their updates, the specific update formula leads to different convergence speeds. 

<mark class="pink">**There are two special cases**:</mark>
1. When $\sigma_t = 0$
    - The random noise term disappears;
    - The trajectory becomes fully deterministic (this is DDIM);
2. When $\sigma\_t=\sqrt{\frac{1 - \alpha_{t-1}}{1 - \alpha_t}} \sqrt{1 - \frac{\alpha_t}{\alpha_{t-1}}}$
    - The process becomes equivalent to a DDPM (a Markov chain);
    - We recover the original DDPM sampling behavior;

It is straightforward to prove the second point:
$$
\begin{aligned}
\sigma\_t&=\sqrt{\frac{1 - \alpha_{t-1}}{1 - \alpha_t}} \sqrt{1 - \frac{\alpha_t}{\alpha_{t-1}}} \quad \quad \color{green}\small{\text{DDIM}} \\\  
&=\sqrt{\frac{1 - {\color{red}\bar{\alpha}_{t-1}}}{1 - {\color{red}\bar{\alpha}_t}}} \underbrace{\sqrt{1 - {\color{red}\alpha_t}}}\_{\sqrt{\beta_t}} \quad \quad \quad \ \color{green}\small{\text{DDPM; notation change from $\alpha_t$ to $\frac{\alpha\_t}{\alpha\_{t-1}}$}} \\\ 
&= \sqrt{\tilde{\beta}_t}
\end{aligned}
$$

So DDIM and DDPM are not completely separate models. DDIM gives us a continuum of samplers, where DDPM and deterministic DDIM are just two endpoints.

| **Feature** |	**DDPM** | **DDIM** |
|-------------|----------|----------|
| Reverse Process | Stochastic | Deterministic or low-noise |
| Sampling Speed | Slow (often 1000+ steps)	$\quad$ | Fast (often 10-50 steps) |
| Supports Step Skipping $\quad \quad$ | ❌ No  |  ✅ Yes |
| Quality | Very high | Almost the same |
| Requires Retraining? | Often yes | No, use same model |

## Summary
Let’s recap the main ideas:

- DDPMs gradually add and remove noise in many small steps, which makes sampling slow but produces very high-quality results.
- DDIMs keep the same noisy distributions $q(\mathbf{x}_t \vert \mathbf{x}\_0)$ but change the reverse process to be non-Markovian and often deterministic.
- By reusing the same noise across steps and introducing “memory” of the original data, DDIMs can take larger jumps along the diffusion trajectory.
- This allows DDIM to use far fewer sampling steps (e.g., 10–50 instead of 1000+) while maintaining similar sample quality.
- Because the training objective is essentially the same, a model trained as a DDPM can usually be used directly for DDIM sampling.

**In short**: *DDIM is a faster way to sample from diffusion models* that you can apply to many existing DDPM models without retraining.

## References
[^Sohl-Dickstein2015]: Sohl-Dickstein, J. et al., 2015. [Deep unsupervised learning using nonequilibrium thermodynamics](https://arxiv.org/abs/1503.03585). *Proceedings of the 32$^{nd}$ International Conference on Machine Learning (ICML)*, PMLR, 37, pp.2256–2265.
[^Ho2020]: Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*, pp. 6840-6851.
[^JSong2020]: Song, J., Meng, C., & Ermon, S., 2020. [Denoising diffusion implicit models](https://arxiv.org/abs/2010.02502). arXiv preprint arXiv:2010.02502.
[^Chan2024]: Chan, S., 2024. [Tutorial on Diffusion Models for Imaging and Vision](https://dl.acm.org/doi/abs/10.1561/0600000112). *Found. Trends. Comput. Graph. Vis.* **16**, 4, 322–471.
