---
title: "Diffusion Models (Part 3): Denoising Diffusion Implicit Models"
description: "A beginner-friendly guide to Denoising Diffusion Implicit Models (DDIMs) — a faster and more efficient alternative to DDPMs."
summary: "A beginner-friendly guide to Denoising Diffusion Implicit Models (DDIMs) — a faster and more efficient alternative to DDPMs."
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

<span style="color: #1E90FF;"><small>[This post is a work in progress -- content will be updated!]</small></span>

[In Part 1]({{< relref "../Intro-Diffusion-Models-part1/index.md" >}}), we explored *Denoising Diffusion Probabilistic Models (DDPMs)*[^Ho2020] and saw how they turn pure noise into meaningful data. 
The process is elegant, but there’s a catch: ***sampling takes a huge of steps, which makes it painfully slow in practice***.

That’s where *Denoising Diffusion Implicit Models (DDIMs)*[^JSong2020] come in. They build on the same core ideas as DDPMs but traverse the noise space far more efficiently, producing high-quality samples in far fewer steps.

In this post, we’ll see how DDIMs work and why they’re so much faster.
We’ll touch on some math, but only enough to develop an intuitive sense of what’s going on.

{{< quote-red >}}
*If you're new to diffusion models, we recommend reading [Part 1]({{< relref "../Intro-Diffusion-Models-part1/index.md" >}}) for helpful background.*
{{< /quote-red >}}

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## Notations
| **Symbols** |	**Meaning** |
|-------------|-------------|
| $\mathbf{x}_0$ | Original clean sample	|
| $\mathbf{x}_t$ | Noisy sample at step $t$	|
| $\mathbf{x}\_{1:T}$ | Sequence of increasingly noisy samples $(\mathbf{x}_1, \dots, \mathbf{x}_T)$	|
| $\\{\beta\_t \in (0, 1)\\}\_{t=1}^T \quad \quad$ | Variance schedule (hyper-parameter)	|
| $\mathcal{N}(\boldsymbol{\mu},\sigma)$ | Normal distribution with mean $\boldsymbol{\mu}$ and variance $\sigma$ |
| $q(\mathbf{x}_i \vert \mathbf{x}_j)$ | Conditional probability distribution of $\mathbf{x}_i$ given $\mathbf{x}_j$ |


## The Problem with DDPMs
Let's do a short re-cap of DDPMs first and explore their disadvantage.

Suppose we have a real sample $\mathbf{x}_0 \sim q(\mathbf{x})$.
In DDPM forward process, we gradually corrupt the data by adding small amounts of Gaussian noise over $T$ steps, producing a sequence of increasingly noisy samples $(\mathbf{x}_1, \dots, \mathbf{x}_T)$.
Recall the [DDPM forward transition]({{< relref "../intro-diffusion-models-part1/index.md#the-forward-process-adding-noise" >}}):
\begin{equation}
q(\mathbf{x}\_t \vert \mathbf{x}\_{t-1}) = \mathcal{N}(\mathbf{x}\_t; \sqrt{1 - \beta\_t} \mathbf{x}\_{t-1}, \beta\_t\mathbf{I})
\label{eq:ddpm-forward}
\end{equation}

Mathematically, we can write each step of the process as:
\begin{equation}
\mathbf{x}\_t = \sqrt{1 - \beta\_t}\mathbf{x}\_{t-1} + \sqrt{\beta\_t}\boldsymbol{\epsilon}\_{t-1} \quad \quad \text{where } \boldsymbol{\epsilon}\_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
\end{equation}

This forms a [*Markov chain*]({{< relref "../intro-diffusion-models-part1/index.md#markov-chain" >}}) where each state $\mathbf{x}\_{t}$ depends only on the immediately preceding state $\mathbf{x}\_{t-1}$ and the noise $\boldsymbol{\epsilon}\_{t-1}$ added at each step. 

Using a [*reparameterization trick*]({{< relref "../intro-diffusion-models-part1/index.md#reparameterization-trick" >}}), the forward process can be written in closed form:
\begin{equation}
q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}\big(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I}\big)
\label{eq:reparameter}
\end{equation}
with $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}\_t = \prod\_{i=1}^t \alpha\_i$.

A <mark class="blue">major benefit</mark> of a Markov chain is that it is simple and has a *memoryless structure*; i.e., the next state can be sampled using only the current one.

There is <mark class="pink">a key limitation</mark>, however.
Because $\beta_t$ must be small to ensure stability, DDPMs require many ($T \gg 0$) tiny increments to reach the fully noised state $\mathbf{x}\_T \sim \mathcal{N}(0,\boldsymbol{I})$, and the reverse process needs to traverse the entire chain one step at a time (i.e., $T$ steps).
This makes generating samples in DDPMs very slow.

<div style="border: 3px solid seagreen; border-radius: 10px; padding: 0px; width: 99%;">
<b style="margin: 0; text-align: center;">
<span style="background: seagreen; display:block; padding:4px; border-radius: 0px;">
    Walking in a forest    
</span>
</b>
<div style="margin-top: 6px;"> </div>

<div style="padding: 0px 12px 8px 12px;text-align: justify;font-style: italic;">

Imagine you are walking through a dense forest. The trees block your view, so you can only see the ground right in front of you. If each step is based only on your current position, you can move only in small, cautious increments because you have no sense of where the trail is going. <b>This is like a Markov chain (memoryless)</b>.

<div style="margin-top: 12px;"> </div>

Now imagine you also carry a map showing where you started and where the trail leads. With this extra “memory”, you can take larger, more confident steps -- even skip intermediate ones -- because you know the general direction. <b>This is like a non-Markovian process: you’re guided not just by the present moment, but by global information</b>.
</div>
<div style="margin-top: -24px;"> </div>
</div>

<br>
<!--Because $\beta\_t$ is small at each timestep, DDPMs require many tiny increments to reach the fully noised state $\mathbf{x}\_T \sim \mathcal{N}(0,\boldsymbol{I}).$-->

DDIMs address the sampling inefficiency of DDPM by formulating <mark class="orange">*non-Markovian*</mark> processes, enabling deterministic and flexible sampling schedules.

><mark class="orange">*A non-Markovian process is a system whose future behavior depends not only on its current state but also on its past history -- meaning the process has memory.*</mark>

{{< quote-blue >}}
**Why memory matters?** <br>
Memory allows a process to retain information from its past, which often influences its future evolution. Non-Markovian structures capture these dependencies, enabling more accurate modeling of system dynamics, smoother trajectories, and better predictions compared to memoryless (Markovian) systems.
{{< /quote-blue >}}

Let’s explore how DDIMs do this!

## Forward Process in DDIM
DDIM starts by reparameterizing the DDPM forward process. Instead of using the coefficient $\alpha_t$ directly, DDIM expresses each step using the ratio $\color{blue}\frac{\alpha_t}{\alpha_{t-1}}$. Thus, Eqn \eqref{eq:ddpm-forward} becomes:
\begin{equation}
q(\mathbf{x}\_t \vert \mathbf{x}\_{t-1}) = \mathcal{N}\left(\mathbf{x}\_t; \sqrt{{\color{blue}\frac{\alpha_t}{\alpha_{t-1}}}} \mathbf{x}\_{t-1},  \big(1-{\color{blue}\frac{\alpha_t}{\alpha_{t-1}}}\big)\mathbf{I}\right)
\end{equation}

This change does not carry any physical meaning but simplifies notation. Under this form, the product term in Eqn \eqref{eq:reparameter} is simplified to:

$$
\bar{\alpha_t}=\prod\_{i=1}^t {\color{blue}\frac{\alpha\_i}{\alpha\_{i-1}}}={\color{red}\alpha_t} \quad \quad ; \text{assuming} \ \alpha_0=1
$$

The forward process in DDIM becomes:
\begin{equation}
q(\mathbf{x}_t \vert \mathbf{x}\_0) = \mathcal{N}\big(\mathbf{x}\_t; \sqrt{{\color{red}\alpha_t}} \mathbf{x}_0, (1 - {\color{red}\alpha_t})\mathbf{I}\big)
\end{equation}

>*<mark>These marginals are identical to those of DDPM, which means a DDPM-trained model can be used directly for DDIM sampling.</mark>*

Again, using the reparameterization trick, we can write:
\begin{equation}
\mathbf{x}\_t = \sqrt{\alpha\_t}\mathbf{x}\_0 + \sqrt{1 - \alpha\_t}\boldsymbol{\epsilon} \quad \quad ; \text{where} \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0,\boldsymbol{I})
\label{eq:6}
\end{equation}

and similarly,
\begin{equation}
\mathbf{x}\_{t-1} = \sqrt{\alpha\_{t-1}}\mathbf{x}\_0 + \sqrt{1 - \alpha\_{t-1}}\boldsymbol{\epsilon}
\label{eq:7}
\end{equation}

The trick here is to replace $\boldsymbol{\epsilon}$ by something so that $\mathbf{x}\_{t-1}$ is no longer $\mathbf{x}\_0$ perturbed by white noise.
To do this, we solve Eqn \eqref{eq:6} for $\boldsymbol{\epsilon}$:

$$
\boldsymbol{\epsilon} = \frac{\mathbf{x}\_t - \sqrt{\alpha\_t} \mathbf{x}\_0}{\sqrt{1 - \alpha\_{t}}}
$$

Substituting into Eqn \eqref{eq:7}, we obtain:
\begin{equation}
\mathbf{x}\_{t-1} = \sqrt{\alpha\_{t-1}}\mathbf{x}\_0 + \sqrt{1 - \alpha\_{t-1}} \color{red}\left(\frac{\mathbf{x}\_t - \sqrt{\alpha\_t} \mathbf{x}\_0}{\sqrt{1 - \alpha\_{t}}} \right)
\label{eq:8}
\end{equation}

This formulation removes the need for explicit sampling from a Gaussian at every timestep as in DDPMs. Instead, $\mathbf{x}_{t-1}$ is now a deterministic function of $\mathbf{x}_t$ and an estimate of $\mathbf{x}_0$.

<div style="border: 3px solid lightsalmon; border-radius: 10px; padding: 0px; width: 99%;">
<h4 style="margin: 0; text-align: center;">
<span style="background: lightsalmon; display:block; padding:4px; border-radius: 0px;">
    Why does this make sampling faster?
</span>
</h4>
<div style="margin-top: 6px;"> </div>

<div style="padding: 0px 12px 8px 12px;text-align: justify;font-style: italic;">
With no new random noise to add or undo, the sampler no longer needs many small corrective steps. Each update becomes stable and predictable, allowing DDIM to skip large portions of the trajectory and generate high-quality samples in far fewer iterations.
</div>
<div style="margin-top: 0px;"> </div>
</div>

<br>

### A family of inference processes
DDIMs generalize the idea further by defining a family $\mathcal{Q}$ of inference processes indexed by a noise vector $\sigma \in \mathbb{R}^T_{\ge 0}$:

\begin{equation}
q_{\sigma}(\mathbf{x}\_{1:T} \vert \mathbf{x}\_0) = q_{\sigma}(\mathbf{x}\_T \vert \mathbf{x}\_0) \prod^T\_{t=2} q_{\sigma}(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0)
\end{equation}
where the terminal distribution is defined as
$$
q_{\sigma}(\mathbf{x}\_T \vert \mathbf{x}\_0) = \mathcal{N}\big(\sqrt{\alpha\_T} \mathbf{x}_0, (1 - \alpha\_T)\mathbf{I}\big)
$$

The forward process is also Gaussian and can be derived from Bayes’ rule:

\begin{equation}
q_{\sigma}(\mathbf{x}\_{t} \vert \mathbf{x}\_{t-1}, \mathbf{x}\_0) = \frac{q_{\sigma}(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0) q_{\sigma}(\mathbf{x}\_{t} \vert, \mathbf{x}\_0)}{q_{\sigma}(\mathbf{x}\_{t-1} \vert \mathbf{x}\_0)}
\label{eq:ddim-forward}
\end{equation}

Unlike the DDPM forward process, the DDIM forward process in Eqn \eqref{eq:ddim-forward} allows each state $\mathbf{x}\_t$ to depend on both $\mathbf{x}\_{t-1}$ and the clean signal $\mathbf{x}\_0$. 

This is a key distinction from DDPM that makes the process *non-Markovian and no longer "memoryless"*.

{{< figure
  src="../../images/ddim_inferences.png"
  alt="Diffusion model"
  caption="Graphical models for diffusion (top) and non-Markovian (bottom) inference models. <small>*Adapted from [Song et al (2020)](https://arxiv.org/abs/2010.02502)*</small>."
>}}

By allowing the process to "remember" $\mathbf{x}\_0$, DDIM introduces additional structure without changing the noisy marginals.
This added flexibility is what enables more efficient (and deterministic) generative trajectories -- an idea we explore in the next section.

## Generative Processes in DDIM
The most important argument in DDIM is: for all time $t$, we want the marginal distribution $q_{\sigma}(\mathbf{x}\_{t} \vert \mathbf{x}\_0)$ to have the same form: 

\begin{equation}
q_{\sigma}(\mathbf{x}_{t} \vert \mathbf{x}\_0) = \mathcal{N}\big(\sqrt{\alpha\_{t}} \mathbf{x}_0, (1 - \alpha\_{t})\mathbf{I}\big)
\label{eq:9}
\end{equation}

The reason is that ultimately we want $q_{\sigma}(\mathbf{x}\_t \vert \mathbf{x}\_0)$ to become pure white noise when $t=T$ and the original sample when $t=0$.
There are many different choices of the reverse transition distribution $q\_{\sigma}(\mathbf{x}_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0)$, but only some of them can ensure that $q\_{\sigma}(\mathbf{x}\_t \vert \mathbf{x}\_0)$ takes the form we want above in Eqn \eqref{eq:9}.

For this purpose, the reverse transition distribution in DDIM is chosen as follows[^JSong2020]:
\begin{equation}
q_{\sigma}(\mathbf{x}_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0) = \mathcal{N}\big(\sqrt{\alpha\_{t-1}}\mathbf{x}\_0 + \sqrt{1 - \alpha\_{t-1}-\sigma_t^2} \left(\frac{\mathbf{x}\_t - \sqrt{\alpha\_t} \mathbf{x}\_0}{\sqrt{1 - \alpha\_{t}}} \right), \sigma_t^2\mathbf{I}\big)
\end{equation}

><mark>*Check out this tutorial[^Chan2024] to see the detailed derivation of this transition distribution.*</mark>

The magnitude of $\sigma_t$ controls how stochastic the forward process is. When $\sigma_t \rightarrow 0$, we reach an extreme case where as long as we observe $\mathbf{x}\_0$ and $\mathbf{x}\_t$ for some $t$, then $\mathbf{x}\_{t-1}$ becomes known and fixed.

### Inference for DDIM
The inference for DDIM is derived based on the transition distribution. Starting with the forward process, if we want to perform the reverse, we will need to find out $\mathbf{x}\_0$ from Eqn \eqref{eq:6}:

\begin{equation}
\underbrace{\mathbf{x}\_t}\_{\text{given}} = \underbrace{\sqrt{\alpha\_t}\mathbf{x}\_0}\_{\text{want to find}} + \underbrace{\sqrt{1 - \alpha\_t}\boldsymbol{\epsilon}}\_{\text{estimated by network}}
\end{equation}

By rearranging the terms:
\begin{equation}
\mathbf{x}\_0 = \frac{1}{\sqrt{\alpha\_t}} \left( \mathbf{x}\_t - \sqrt{1 - \alpha\_t}\boldsymbol{\epsilon} \right) \\\
\end{equation}

This can be approximated by a neural network:
\begin{equation}
{\color{red}f\_{\theta}^{(t)}(\mathbf{x}\_t)} = \frac{1}{\sqrt{\alpha\_t}} \left( \mathbf{x}\_t - \sqrt{1 - \alpha\_t} \cdot {\color{red}\boldsymbol{\epsilon}\_{\theta}^{(t)}(\mathbf{x}\_t)} \right)
\end{equation}

There are two new terms in this equation. The first one is $\color{red}\boldsymbol{\epsilon}\_{\theta}^{(t)}(\mathbf{x}\_t)$ which replaces $\boldsymbol{\epsilon}$. It is the estimate of the noise based on the current input $\mathbf{x}\_t$.
The second term is the denoised estimator $\color{red}f\_{\theta}^{(t)}(\mathbf{x}\_t)$, which is a prediction of the true signal $\mathbf{x}\_0$ given $\mathbf{x}\_t$.

Back to the transition distribution $q\_{\sigma}(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, \mathbf{x}\_0)$, if we do not have access to $\mathbf{x}\_0$, we can replace it with $\color{red}f\_{\theta}^{(t)}(\mathbf{x}\_t)$:

\begin{equation}
\begin{aligned}
p\_{\theta}(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t) &= q\_{\sigma}(\mathbf{x}\_{t-1} \vert \mathbf{x}\_t, {\color{red}f\_{\theta}^{(t)}(\mathbf{x}\_t)}) \\\
&= \mathcal{N}\big(\sqrt{\alpha\_{t-1}} \left( \frac{\mathbf{x}\_t - \sqrt{1 - \alpha\_t}\color{red}\boldsymbol{\epsilon}\_{\theta}^{(t)}(\mathbf{x}\_t)}{\sqrt{\alpha\_t}} \right) + \sqrt{1 - \alpha\_{t-1}-\sigma_t^2} \cdot {\color{red}\boldsymbol{\epsilon}\_{\theta}^{(t)}(\mathbf{x}\_t)}, \sigma_t^2\mathbf{I}\big)
\end{aligned}
\label{eq:reverse-eq}
\end{equation}

For the special case where $t=1$, we define:
$$
p\_{\theta}(\mathbf{x}\_{0} \vert \mathbf{x}\_1) = \mathcal{N}\big( {\color{red}f\_{\theta}^{(1)}(\mathbf{x}\_1)}, \sigma_1^2\mathbf{I} \big)
$$
so that the reverse process is supported everywhere.


<!--The generative process in DDIM is considered as the approximation to the reverse process; since of the forward process has $T$ steps, the generative process is also forced to sample $T$ steps. -->
<!--However, as the denoising objective does not depend on the specific forward procedure as long as $q_{\sigma}(\mathbf{x}_{t} \vert \mathbf{x}\_0)$ is fixed, we may also consider forward processes with lengths smaller than $T$, which accelerates the corresponding generative processes without having to train a different model[^JSong2020].-->

### Accelerated sampling
Traditionally, a full generative trajectory requires iterating through all $T$ diffusion steps. However, DDIM introduces an important insight: the sampling schedule is not tied to the forward noising schedule. This flexibility allows us to define a reduced sequence of timesteps $\tau = \\{\tau_1,\dots,\tau_S\\}$ and run the reverse process only at those points.

{{< figure
  src="../../images/ddim_accelerated.png"
  alt="Diffusion model"
  caption="Graphical model for accelerated generation, where $\tau=[1, 3]$. <small>*Image source: [Song et al (2020)](https://arxiv.org/abs/2010.02502)*</small>."
  width=80%
>}}

In other words, instead of stepping through all ($T$) noise levels, we can "jump" between them while still following a coherent generative path. Each DDIM update remains stable and predictable because it is derived from the deterministic structure of the transition.
When the length of this sampling trajectory $S \ll T$, we can achieve a significant increase in computational efficiency while still preserving high sample quality.

This is the foundation of accelerated sampling in DDIM and one of the main reasons it is widely used in practice.

<!--Let us consider the forward process as defined not on all the latent variables $\mathbf{x}\_{1:T}$ , but on a subset $\{\mathbf{x}\_{\tau_1},...,\mathbf{x}\_{\tau_S}\}$, where $\tau$ is an increasing sub-sequence of $[1,...,T]$ of length $S$. In particular, we define the sequential forward process over $\mathbf{x}\_{\tau_1},...,\mathbf{x}\_{\tau_S}$ such that $q_{\sigma}(\mathbf{x}_{\tau_i} \vert \mathbf{x}\_0)=\mathcal{N}\big(\sqrt{\alpha\_{\tau_i}}\mathbf{x}_0), (1-\alpha\_{\tau_i})\mathbf{I}\big)$ matches the "marginals". The generative process now samples latent variables according to reversed ($\tau$), which we term (sampling) trajectory. When the length of the sampling trajectory is much smaller than $T$, we may achieve significant increases in computational efficiency due to the iterative nature of the sampling process.-->


## DDIM vs DDPM: Key Differences
From $p_\theta(x_{1:T})$ in Eqn \eqref{eq:reverse-eq}, we can generate a sample $x_{t-1}$ from a sample $x_{t}$ via:

\begin{equation}
\text{DDIM:} \quad \mathbf{x}\_{t-1} = \sqrt{\alpha\_{t-1}} \underbrace{\left(\frac{\mathbf{x}\_t - \sqrt{1 - \alpha\_t} {\color{red}\boldsymbol{\epsilon}\_{\theta}^{(t)}(\mathbf{x}\_t)} }{\sqrt{\alpha\_t}}\right)}\_{\text{predicted} \ x\_0} + \underbrace{\sqrt{1 - \alpha\_{t-1} - \sigma\_t^2} \cdot {\color{red}\boldsymbol{\epsilon}\_{\theta}^{(t)}(\mathbf{x}\_t)}}\_{\text{direction pointing to } \mathbf{x}\_t} + \underbrace{\sigma\_t \boldsymbol{\epsilon}\_t}\_{\text{random noise}} 
\label{eq:sample-eq-gen}
\end{equation}

It would be helpful to compare this equation with the DDPM equation (note that $\alpha\_t$ in DDPM is different from the one in DDIM):
\begin{equation}
\text{DDPM:} \quad \quad \quad \mathbf{x}\_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}\_t}} \boldsymbol{\epsilon}\_{\theta}^{(t)}(\mathbf{x}\_t) \right) + \sigma\_t \boldsymbol{\epsilon}\_t
\end{equation}

The main difference between DDPM and DDIM is subtle. While they both use $\mathbf{x}\_t$ and $\boldsymbol{\epsilon}\_{\theta}^{(t)}(\mathbf{x}\_t)$ in their updates, the specific update formula makes a different convergence speed. 

**Special cases** -- Note that forward process in DDIM becomes:
- deterministic (except for $t=1$) when $\sigma\_t=0, \forall t$ 
- Markovian (and the generative process becomes a DDPM) when: 
$$\sigma\_t=\sqrt{\frac{(1 - \alpha_{t-1})}{(1 - \alpha_t)} (1 - \frac{\alpha_t}{\alpha_{t-1}})}$$


| **Feature** |	**DDPM** | **DDIM** |
|-------------|----------|----------|
| Reverse Process | Stochastic | Deterministic or low-noise |
| Sampling Speed | Slow (1000+ steps)	| Fast (10-50 steps) |
| Supports Step Skipping $\quad \quad$ | ❌ No  |  ✅ Yes |
| Quality | Very High | Almost the same |
| Requires Retraining? | Yes, for new schedules $\quad$ | No, use same model |

## Summary
- DDIM is a faster alternative to DDPM for generation
- Uses a deterministic reverse process, reducing randomness
- Supports fewer steps without hurting quality much
- Works with existing pre-trained models

## References
[^Sohl-Dickstein2015]: Sohl-Dickstein, J. et al., 2015. [Deep unsupervised learning using nonequilibrium thermodynamics](https://arxiv.org/abs/1503.03585). *Proceedings of the 32$^{nd}$ International Conference on Machine Learning (ICML)*, PMLR, 37, pp.2256–2265.
[^Ho2020]: Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*, pp. 6840-6851.
[^JSong2020]: Song, J., Meng, C., & Ermon, S., 2020. [Denoising diffusion implicit models](https://arxiv.org/abs/2010.02502). arXiv preprint arXiv:2010.02502.
[^Chan2024]: Chan, S., 2024. [Tutorial on Diffusion Models for Imaging and Vision](https://dl.acm.org/doi/abs/10.1561/0600000112). *Found. Trends. Comput. Graph. Vis.* **16**, 4, 322–471.
