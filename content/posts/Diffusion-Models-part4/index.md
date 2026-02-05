---
title: "Diffusion Models (Part 4): Precipitation Estimation from Satellite Imagery"
description: "AI is transforming how we observe and understand Earth’s water cycle."
summary: "From orbit to rainfall: how AI helps us understand Earth’s water cycle."
date: 2026-01-06
tags: ["Diffusion Model", "Precipitation Retrieval", "Satellites", "GPM"]
author: "Clement Guilloteau, Efi Foufoula-Georgiou, Phong Le"
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
    image: "https://live.staticflickr.com/65535/54004674805_1d5b8a504a_b.jpg"
    caption: "Image source: [Geostationary Operational Environmental Satellites (GOES) - R Series](https://gpm.nasa.gov/missions/GPM)."
    relative: true # when using page bundles set this to true
    hidden: false          # don't hide globally
    hiddenInList: true     # hide in list pages
    hiddenInSingle: false  # show inside post
editPost:
    URL: "https://github.com/ess-aiml/blogs/blob/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

<span style="color: #1E90FF;"><small>[This post is a work in progress -- content will be updated!]</small></span>

From orbit, satellites now capture a constant stream of images and signals that reveal the movement of clouds, moisture, and storms across the planet. What has long remained elusive is how to translate these observations into reliable estimates of ___precipitation___ -- one of the most important and difficult variables in weather and climate science. 

In this Part&nbsp;4, we explore how generative Artificial Intelligence (AI) methods, particularly diffusion models, are beginning to close this gap.

{{< quote-red >}}
*Readers unfamiliar with diffusion models may wish to begin with reading [Part&nbsp;1]({{< relref "../Intro-Diffusion-Models-part1/index.md" >}}) and [Part&nbsp;2]({{< relref "../Intro-Diffusion-Models-DDIM-part2/index.md" >}}), which introduce the core concepts behind these generative approaches.*
{{< /quote-red >}}

</br>
<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>
</br>

Precipitation includes all forms of water that fall from the atmosphere to Earth -- *rain, snow, sleet, and hail*. Through this process, water stored in clouds returns to the land surface, completing a vital stage of the <mark class="green">[water cycle](https://www.noaa.gov/education/resource-collections/freshwater/water-cycle)</mark>.

><mark class="green">___Definition__: The water cycle describes the continuous movement of water within the Earth and atmosphere._</mark>

<!--This cycle helps sustain life on Earth. -->
Precipitation sustains ecosystems, refills water bodies, supports agriculture, and gradually reshapes landscapes. At the same time, it can pose serious hazards. Heavy rainfall may trigger floods or landslides, while prolonged dry periods can lead to drought, water shortages, and crop failure.

Despite its everyday presence, precipitation is surprisingly difficult to measure accurately over large areas. 
Rainfall can be highly uneven and evolve rapidly, with intense downpours confined to narrow regions while nearby areas remain dry. Capturing this transient and spatially patchy behavior is one of the key challenges in observing Earth’s atmosphere.

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## Tracking Precipitation
To monitor precipitation, we rely on a combination of observational systems, each offering a different perspective on atmospheric processes.

### Weather Radars
Ground-based _weather radars_ are among the most direct and precise instruments for observing precipitation. 
They work by sending out pulses of *radio waves* -- much like a flashlight scanning the sky -- that scatter off raindrops, snowflakes, or hailstones.
By analyzing the returning signal, weather radars can estimate where precipitation is falling and how intense it is. 

Modern weather radars typically use two key techniques:
- <mark class="pink">__Dual polarization__</mark>: Radio waves oscillate in both *horizontal and vertical* directions. The return signals provide more details about the size and shape of precipitation particles, allowing meteorologists to better distinguish between rain, snow, and hail.

- <mark class="blue">__Doppler effect__</mark>: Doppler radars can measure small shifts in signal frequency caused by particle motion. This allows us to track storm movement and detect rotating winds inside severe weather systems.

<!--This effect allows meteorologists to measure the speed of precipitation particles, helping them track storm motion and infer wind patterns inside weather systems.-->
<!--Because of their accuracy and fine spatial detail, radars are often considered the “gold standard” for precipitation measurements.-->

{{< figure
  align=center
  src="../../images/doppler_radar.png"
  alt="Doppler radar"
  caption="Doppler weather radars are remote sensing instruments that transmit electromagnetic pulses into the atmosphere and analyze the returned signals after they scatter from precipitation. <small>Image adapted from [Amanda Montanez](https://www.scientificamerican.com/article/how-doppler-radar-lets-meteorologists-predict-weather-and-save-lives/).</small>"
  width=100%
>}}

Radar systems can also be mounted on satellites, but taking this technology to the space comes with major challenges:
- **Limited coverage**: Weather radars have a range of a few hundred kilometers at most. From space, this means spaceborne radars must fly in [Low Earth Orbits (LEOs)](https://en.wikipedia.org/wiki/Low_Earth_orbit), which restricts their fields of view. 
- **High cost**: Radars are heavy instruments and require a lot of power, meaning only the largest and most expensive satellite platforms can carry them. 

As a result, only a handful of satellite radar missions -- such as [TRMM](https://gpm.nasa.gov/missions/trmm) and [GPM](https://gpm.nasa.gov/) -- have ever been launched. 
While these missions provide invaluable measurements, they are not enough to continuously monitor precipitation across the entire globe or track fast-moving storms at every hour of the day.
<!--These few satellites are not enough to monitor precipitation around the globe or track fast-moving storms at every hour of the day.-->

### Satellite Imagers
Satellite imagers offer a complementary -- *and far more frequent* -- view of Earth’s atmosphere. Instead of actively transmitting signals, these instruments measure electromagnetic radiation naturally emitted or reflected by clouds and the surface, primarily in the ___infrared and microwave___ parts of the spectrum. 

These measurements contain indirect information about cloud structure and water content, which can be linked to precipitation.

Compared to spaceborne radars, satellite imagers provide much higher temporal coverage:

- <mark class="pink">**Infrared imagers**</mark>: Positioned in geostationary orbit, they can monitor an entire hemisphere every 5 to 30 minutes.
- <mark class="gray">**Microwave imagers**</mark>: These satellites orbit relatively close to Earth, and when data from multiple satellites are combined, can provide near-global snapshots every few hours

This rapid sampling makes satellite imagers essential for tracking the life cycle of cloud systems and filling the large spatial and temporal gaps left by radar observations.

{{< figure
  align=center
  src="../../images/Imager-remote-sensing.webp"
  alt="satellite imagers"
  caption="The Advanced Baseline Imager (ABI) on the GOES-R satellites captures energy reflected and emitted from Earth, helping us monitor clouds and weather from space. <small>Image source: [NOAA](https://www.nesdis.noaa.gov/news/transforming-energy-imagery-how-satellite-data-becomes-stunning-views-of-earth)</small>"
  width=90%
>}}

At any given moment, the collection of radiance measurements across multiple channels forms a distinctive electromagnetic signature of the clouds. This spectral "fingerprint" can be used to infer whether precipitation is occurring, what type it is, and how intense it may be. 

Translating these indirect signals into accurate precipitation maps, however, remains a major challenge -- one that has long pushed the limits of traditional methods. 

For decades, satellite-based precipitation retrievals have relied on combinations of physical models and statistical relationships. One fundamental problem is that similar satellite signatures can correspond to very different rainfall outcomes, making uncertainty an unavoidable part of the problem.

This is where AI begins to play a transformative role.

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## AI for Precipitation Estimation from Satellites
Recently, [Guilloteau and colleagues](https://efi.eng.uci.edu/) at the University of California, Irvine introduced **DifERS**[^Guilloteau:2025] (**Dif**fusion-Based **E**nsemble **R**ainfall estimation from **S**atellites), a generative AI model designed to estimate precipitation from satellite data while explicitly representing uncertainty. 

Rather than producing a single "best guess" precipitation map, DifERS generates _an ensemble of physically realistic outcomes_, all consistent with the observed satellite measurements. This probabilistic approach allows the model to capture the inherent uncertainty of satellite-based rainfall retrievals -- providing not only precipitation estimates but also a quantitative measure of confidence.

In the sections that follow, we take a closer look at how DifERS works, unpacking the model framework and exploring how diffusion-based AI makes this new generation of satellite precipitation estimates possible.

<center> <span style="letter-spacing: 0.5rem;">• • •</span> </center>

### Notations
| **Symbols** |	**Meaning** |
|-------------|-------------|
| $K \in \mathbb{N}_{>0}$ | Number of diffusion iterations to reach pure noise ($K$=1000) |
| $\mathbf{x}_0 \in \mathbb{R}^{64 \times 64}$ | Precipitation image from training (original clean sample) |
| $\mathbf{x}_k \in \mathbb{R}^{64 \times 64}$ | Noisy sample at iteration $k$	|
| $\mathbf{z} \in \mathbb{R}^{64 \times 64 \times D}$ | Corresponding satellite measurements, with $D$=23 <small>(See note below)</small>|
| $\\{\beta\_k \in (0, 1)\\}\_{k=1}^K \quad \quad$ | Variance (noise) schedule	|
| $p_{\theta}(\mathbf{x}_0 \vert \mathbf{z})$ | probability distribution of precipitation image $\mathbf{x}_0$ conditioned on the information contained in $\mathbf{z}$	|

<center> <span style="letter-spacing: 0.5rem;">• • •</span> </center>

### Data
DifERS combines passive microwave and infrared satellite observations, which provide complementary information on cloud and precipitation structure:

1. **Special Sensor Microwave Imager/Sounder (SSMI/S)**: 
    - Onboard the [DMSP satellites](https://www.ncei.noaa.gov/products/satellite/defense-meteorological-satellite-program) (F-16, F-17, and F-18)
    - Measures (24 channels) polarized microwave radiances at the top of the atmosphere between 19 and 183 GHz
    - DifERS uses the 10 channels most relevant (and available) for precipitation retrieval
2. **Advanced Baseline Imager (ABI)**: 
    - Onboard NOAA’s [GOES-R geostationary satellites](https://science.nasa.gov/mission/goes/)
    - Provides imaging from geostationary orbit in the visible and infrared domains
    - DifERS uses the thermal infrared 10.3&nbsp;$\mu$m channel
      
In addition to satellite observations, DifERS relies on ground-based reference precipitation data obtained from NOAA’s [Multi-Radar Multi-Sensor (MRMS)](https://www.nssl.noaa.gov/projects/mrms/) system for model training and evaluation.

<div style="border: 3px solid  #b39ddb; border-radius: 16px; padding: 4px 16px; margin: 1em auto; width: 100%;">
<b>Note on data processing:</b>
<ul>
<li style="padding-bottom: 4px;"> All radiometric images from SSMI/S and ABI are projected onto a common regular $5\times5$ km$^2$ spatial grid;
<li style="padding-bottom: 4px;"> MRMS precipitation fields are remapped onto the same $5\times5$ km$^2$ grid and temporally aggregated to hourly resolution;
<li style="padding-bottom: 4px;"> Satellite measurements, $\mathbf{z}$, consist of 10 SSMI/S channels acquired at time $t$ and 13 single-band ABI images (10.3&nbsp;$\mu$m) spanning from $t-30$ to $t+30$ min ($\Delta t = 5$&nbsp;min);
<li style="padding-bottom: 4px;"> As a result, each precipitation image $\mathbf{x}_0$ and its corresponding satellite measurement $\mathbf{z}$ are collocated and cover a spatial extent of $320\times320$ km$^2$.
</ul>
</div>

{{< figure
  align=center
  src="../../images/diff4_brightness_temp.png"
  alt="Brightness temperatures from ABI and SSMI/S"
  caption="**Passive microwave and infrared brightness temperatures measured at the top of the atmosphere.**  (Top, left) Brightness temperature at 92 GHz from SSMI/S onboard DMSP-F17. (Top, right) Stacked SSMI/S brightness temperature for all channels over a 320$\times$320 km subset of the study domain. (Bottom, left) Brightness temperature at 10.3 $\mu m$ from ABI onboard GOES-16 at 13:10 UTC (corresponding to the time of overpass of the DMSP-F17 satellite). (Bottom, right) Time series of ABI brightness temperature fields at 10.3 $\mu m$ from 12:40 to 13:40 UTC over a 320$\times$320 km subset of the study domain. The green rectangle on the left panel delineates the study domain. <small>Image source: [Guilloteau et al., (2025)](https://ieeexplore.ieee.org/abstract/document/10912662).</small>"
  width=100%
>}}

<center> <span style="letter-spacing: 0.5rem;">• • •</span> </center>

### Model Framework

DifERS builds on the denoising diffusion probabilistic model (DDPM)[^Sohl-Dickstein2015]<sup>,</sup>[^Ho2020] framework, which generates complex patterns by learning how to gradually remove noise. The basic idea is intuitive: if we know how real precipitation images look once they have been slightly corrupted by noise, we can learn to reverse that process and recover realistic rainfall fields. Unlike the unconditional diffusion models described in Part 1, DifERS is explicitly conditioned on satellite observations.

Let $\mathbf{x}_0$ represent a true precipitation image drawn from the training data, and let $\mathbf{z}$ denote the corresponding satellite measurements. The goal of DifERS is to learn the conditional distribution

$$
p_{\theta}(\mathbf{x}_0 \mid \mathbf{z}),
$$

which describes how likely a particular precipitation field is, given what the satellites observe.

__Forward Diffusion Process__: </br>
During training, DifERS applies a controlled corruption process to the precipitation field $\mathbf{x}_0$. Gaussian noise is added incrementally over $K$ diffusion steps, producing a sequence of increasingly noisy images $(\mathbf{x}_1, \dots, \mathbf{x}_K)$. Early steps preserve much of the original spatial structure, while later steps progressively erase fine-scale features, until the image becomes nearly indistinguishable from random noise. 

This forward diffusion process serves two purposes. First, it defines a smooth path from realistic precipitation patterns to a simple, well-understood distribution—an approximately isotropic Gaussian. Second, it provides the training signal for the model: at each step, DifERS learns how much noise was added and how that noise distorts precipitation structures. Importantly, this forward process is fixed and purely probabilistic; it does not depend on satellite data and involves no learning.


__Reverse Denoising Process__: </br>
At inference time, the model runs this process in reverse. It begins with pure Gaussian noise and removes it gradually over $K$ steps. At each step, the model uses the satellite observations ($\mathbf{z}$) to guide the denoising, determining which spatial patterns are most consistent with the available measurements.

As noise is removed, coherent precipitation structures gradually emerge, evolving from diffuse patterns into realistic rainfall fields that respect both the satellite information and the learned statistics of precipitation. Because this reverse process is stochastic, each run produces a slightly different outcome. Repeating the process therefore generates an ensemble of plausible precipitation realizations rather than a single deterministic estimate.

Once trained, DifERS can generate precipitation maps at any location and time where ABI and SSMI/S observations are available, delivering both rainfall estimates and their associated uncertainty within a single probabilistic framework.

{{< figure
  align=center
  src="../../images/diff4_DifERS_architeture.png"
  alt="DifERS architecture"
  caption="**Schematic representation of the DifERS architecture**. <small>Image source: [Guilloteau et al., (2025)](https://ieeexplore.ieee.org/abstract/document/10912662)</small>"
  width=100%
>}}

<center> <span style="letter-spacing: 0.5rem;">• • •</span> </center>

### Interpreting the Ensemble

For a single set of satellite observations, DifERS typically generates an ensemble of 128 precipitation maps. Each realization represents an equally probable realization of the true precipitation field, given the SSMI/S and ABI measurements and their associated uncertainties. 

A common way to summarize the ensemble is through the ensemble mean, i.e. the average of all 128 realizations. It is the estimate with the lowest magnitude of the errors, on average. The ensemble mean is however always a conservative estimate. By averaging across many possible realizations, extreme values tend to be reduced.

From the measurements of the ABI and SSMI/S instruments the likelihood of localized extremes precipitation intensities (exceeding 40 mm/h) to occur can be estimated; the information provided by the instruments is however insufficient to pinpoint the exact location of these extreme intensities at 5 km resolution. The extremes intensities often occur at different locations across the 128 DifERS realizations. For this reason, the ensemble mean is generally a spatially smooth field, with dimmed extreme intensities and small variations of the expected precipitation intensity across adjacent pixels. 

The 128 individual realizations, while they are not exact representation of the truth (the high-resolution spatial patterns and the location of the extremes being partially randomized during the generation process), are all realistic precipitation maps, which reproduce the intensity range of the true precipitation maps and the sharp spatial variations of intensity. 
Any of the 128 realizations (or all of them) may for example be used as input of an hydrological to simulate realistic runoff and streamflow.

{{< figure
  align=center
  src="../../images/diff4_precip_field.png"
  alt="DifERS architecture"
  caption="Examples of precipitation fields generated by DifERS."
  width=100%
>}}

<center> <span style="letter-spacing: 0.5rem;">• • •</span> </center>

### Evaluation of DifERS precipitation intensity maps’ accuracy and realism
To evaluate how realistic the DifERS precipitation maps are, they compared are with the gauge-radar derived precipitation maps over the training area. Of particular interest is the statistical distribution of precipitation intensities across multiple scales.
The moments of order 2 to 4 of the distributions (variance, skewness and kurtosis) are compared across multiple spatial scales from 5 to 160 km and the DifERS fields are found to match the gauge-radar reference much more closely than satellite-derived estimates from legacy algorithms. 

The location accuracy of precipitation features in the DifERS maps is evaluated through the spatial coherence between the DifERS ensemble mean and the gauge-radar truth, computed at different wavelengths in the Fourier domain. Again, in terms of spatial coherence with the truth, DifERS surpasses legacy algorithms at all wavelengths.

This demonstrates the ability of a deep generative diffusion model like DiFERS to generate realistic precipitation intensity maps properly constrained by measurements form satellite infrared and microwave imagers.

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## Summary
- Passive microwave and infrared satellite imagers provide complementary but incomplete constraints on precipitation structure.
- Diffusion-based generative models naturally represent retrieval uncertainty by producing ensembles rather than single deterministic estimates.
- DifERS leverages this framework to generate physically realistic, high-resolution precipitation intensity maps that are jointly constrained by satellite infrared and microwave observations.

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## References

[^Guilloteau:2025]:Guilloteau et al., [A Generative Diffusion Model for Probabilistic Ensembles of Precipitation Maps Conditioned on Multisensor Satellite Observations](https://ieeexplore.ieee.org/abstract/document/10912662), *IEEE Trans. Geosci. Remote Sens.*, **63**, 1--15, (2025).
[^Sohl-Dickstein2015]: Sohl-Dickstein, J. et al., 2015. [Deep unsupervised learning using nonequilibrium thermodynamics](https://arxiv.org/abs/1503.03585). *Proceedings of the 32$^{nd}$ International Conference on Machine Learning (ICML)*, PMLR, 37, pp.2256–2265.
[^Ho2020]: Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*, pp. 6840-6851.
