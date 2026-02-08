---
title: "Diffusion Models (Part 4): Precipitation Estimation from Satellite Imagery"
description: "AI is transforming how we observe and understand Earth’s water cycle."
summary: "From orbit to rainfall: how AI helps us understand Earth’s water cycle."
date: 2026-02-06
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

Precipitation includes all forms of water that fall from the atmosphere to Earth -- *rain, snow, sleet, and hail*. Through this process, water stored in clouds returns to the land surface, completing a vital stage of the [_water cycle_](https://www.noaa.gov/education/resource-collections/freshwater/water-cycle).

><mark class="green">___Definition__: The water cycle describes the continuous movement of water within the Earth systems and atmosphere._</mark>

<!--This cycle helps sustain life on Earth. -->
Precipitation sustains ecosystems, refills water bodies, supports agriculture, and gradually reshapes landscapes. At the same time, it can pose serious hazards. Heavy rainfall may trigger floods or landslides, while prolonged dry periods can lead to drought, water shortages, and crop failure.

Despite its presence everyday, precipitation is surprisingly hard to measure accurately over large areas. 
Rainfall can be highly uneven and evolve rapidly, with intense downpours confined to narrow regions while nearby areas remain dry. Capturing this transient and spatially patchy behavior is one of the central challenges in observing Earth’s atmosphere.

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## Tracking Precipitation
To monitor precipitation, scientists rely on multiple observational systems, each offering a different perspective on atmospheric processes.

### Weather Radars
Ground-based _weather radars_ are among the most direct and precise instruments for observing precipitation. 
They work by sending out pulses of *radio waves* -- much like a flashlight scanning the sky -- that scatter off raindrops, snowflakes, or hailstones.
By analyzing the returning signal, weather radars can estimate where precipitation is falling and how intense it is. 

Modern weather radars typically reply on two key techniques:
- <mark class="pink">__Dual polarization__</mark>: Radio waves oscillate in both *horizontal and vertical* directions. The return signals provide more details about the size and shape of precipitation particles, allowing meteorologists to better distinguish between rain, snow, and hail.

- <mark class="blue">__Doppler effect__</mark>: Doppler radars can measure small _shifts in signal frequency_ caused by particle motion. This allows us to track storm movement and detect rotating winds inside severe weather systems.
<!--Because of their accuracy and fine spatial detail, radars are often considered the “gold standard” for precipitation measurements.-->

{{< figure
  align=center
  src="../../images/doppler_radar.png"
  alt="Doppler radar"
  caption="Doppler weather radars are remote sensing instruments that transmit electromagnetic pulses into the atmosphere and analyze the returned signals after they scatter from precipitation. <small>Image adapted from [Amanda Montanez](https://www.scientificamerican.com/article/how-doppler-radar-lets-meteorologists-predict-weather-and-save-lives/).</small>"
  width=100%
>}}

Radar systems can also be mounted on satellites, but taking this technology to the space comes with major challenges:
- **Limited coverage**: Weather radars have a range of a few hundred kilometers at most. From space, this means spaceborne radars must operate in [Low Earth Orbits (LEOs)](https://en.wikipedia.org/wiki/Low_Earth_orbit), which restricts their fields of view. 
- **High cost**: Radars are heavy instruments and require a lot of power, meaning only the largest and most expensive satellite platforms can carry them. 

As a result, only a handful of satellite radar missions (_e.g._, [TRMM](https://gpm.nasa.gov/missions/trmm) and [GPM](https://gpm.nasa.gov/)) have ever been launched. 
While these missions provide invaluable measurements, they are not enough to continuously monitor precipitation across the entire globe or track fast-moving storms at every hour of the day.
<!--These few satellites are not enough to monitor precipitation around the globe or track fast-moving storms at every hour of the day.-->

<center> <span style="letter-spacing: 0.5rem;">• • •</span> </center>

### Satellite Imagers
Satellite imagers offer a complementary -- *and far more frequent* -- view of Earth’s atmosphere. Instead of actively transmitting signals, they measure electromagnetic radiation naturally emitted or reflected by clouds and the surface, primarily in the ___infrared and microwave___ parts of the spectrum. 
These measurements provide indirect information about cloud temperature, structure, and water content -- clues that can be linked to precipitation processes.

Compared to spaceborne radars, satellite imagers offer much higher temporal coverage:

- <mark class="pink">**Infrared imagers**</mark>, positioned in geostationary orbit, can monitor an entire hemisphere every 5 to 30 minutes.
- <mark class="gray">**Microwave imagers**</mark>, orbiting relatively close to Earth, can provide near-global coverage every few hours when data from multiple satellites are combined.

This rapid sampling makes satellite imagers essential for tracking the life cycle of cloud systems and filling the spatial and temporal gaps left by radar observations.

{{< figure
  align=center
  src="../../images/Imager-remote-sensing.webp"
  alt="satellite imagers"
  caption="The Advanced Baseline Imager (ABI) on the GOES-R satellites captures energy reflected and emitted from Earth, helping us monitor clouds and weather from space. <small>Image source: [NOAA](https://www.nesdis.noaa.gov/news/transforming-energy-imagery-how-satellite-data-becomes-stunning-views-of-earth)</small>"
  width=90%
>}}

At any given moment, the collection of radiance measurements across multiple channels forms a distinctive electromagnetic signature of the clouds. This spectral "_fingerprint_" can be used to infer whether precipitation is occurring, what type it is, and how intense it may be. 

However, translating these indirect signals into accurate precipitation maps remains a major challenge -- one that has long pushed the limits of traditional methods. 

{{< quote-blue >}}
_For decades, satellite-based precipitation retrievals have relied on combinations of physical models and statistical relationships. One fundamental problem is that similar satellite signatures can correspond to very different rainfall outcomes, making _uncertainty_ an unavoidable part of the problem_.
{{< /quote-blue >}}

This is where AI begins to play a transformative role.

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## AI for Precipitation Estimation
Recently, [Guilloteau and colleagues](https://efi.eng.uci.edu/) at the University of California, Irvine introduced **DifERS**[^Guilloteau:2025] (**Dif**fusion-Based **E**nsemble **R**ainfall estimation from **S**atellites). This generative AI model represents a fundamental shift in how we retrieve precipitation: _rather than providing a single "best guess" estimate, it generates an  ensemble of physically realistic possibilities_. 

In the sections that follow, we examine how DifERS works and why diffusion-based generative modeling enables a new generation of satellite-based precipitation estimates.

<center> <span style="letter-spacing: 0.5rem;">• • •</span> </center>

### Notations
| **Symbols** |	**Meaning** |
|-------------|-------------|
| $\mathbf{x}_0 \in \mathbb{R}^{64 \times 64} \quad \quad $ | Precipitation image from training (original, clean sample) |
| $\mathbf{z} \in \mathbb{R}^{64 \times 64 \times D}$ | Corresponding satellite measurements, with $D$=23 <small>(See Data Processing)</small>|
| $p_{\theta}(\mathbf{x}_0 \vert \mathbf{z})$ | Probability distribution of precipitation image $\mathbf{x}_0$ conditioned on the information contained in $\mathbf{z}$	|
| $\theta$ | Set of trainable model parameters in DifERS |

<center> <span style="letter-spacing: 0.5rem;">• • •</span> </center>

### The Algorithm
DifERS employs a conditional diffusion model based on DDPMs[^Sohl-Dickstein2015]<sup>,</sup>[^Ho2020] to produce precipitation intensity maps from combined measurements of two satellite imagers:
- __Special Sensor Microwave Imager/Sounder (SSMI/S)__: onboard the [DMSP satellites](https://www.ncei.noaa.gov/products/satellite/defense-meteorological-satellite-program) 
- __Advanced Baseline Imager (ABI)__: onboard the [GOES-R satellite series](https://science.nasa.gov/mission/goes/)

{{< figure
  align=center
  src="../../images/diff4_brightness_temp.png"
  alt="Brightness temperatures from ABI and SSMI/S"
  caption="**Passive microwave and infrared brightness temperatures (TB) measured at the top of the atmosphere.**  (Top, left) Brightness temperature at 92 GHz from SSMI/S onboard DMSP-F17. (Top, right) Stacked SSMI/S brightness temperature for all channels over a 320$\times$320 km subset of the study domain. (Bottom, left) Brightness temperature at 10.3 $\mu m$ from ABI onboard GOES-16 at 13:10 UTC (corresponding to the time of overpass of the DMSP-F17 satellite). (Bottom, right) Time series of ABI brightness temperature fields at 10.3 $\mu m$ from 12:40 to 13:40 UTC over a 320$\times$320 km subset of the study domain. The green rectangle on the left panel delineates the study domain. <small>Image source: [Guilloteau et al., (2025)](https://ieeexplore.ieee.org/abstract/document/10912662).</small>"
  width=100%
>}}

The algorithm takes all available SSMI/S and ABI measurements, $\mathbf{z} \in \mathbb{R}^{64 \times 64 \times 23} $, within a one-hour time frame over a 320&nbsp;km$\times$320&nbsp;km domain as inputs and produces <mark>_an ensemble of 128 possible maps_</mark> of the  hourly-averaged precipitation intensity at a resolution of 5 km. 

The target precipitation intensity fields, $\mathbf{x}_0 \in \mathbb{R}^{64 \times 64}$, used during the training of DifERS are derived from ground radar and gauge measurements (NOAA's [MRMS](https://www.nssl.noaa.gov/projects/mrms/)) over the United States, in areas where the radar and gauge coverage allows for spatially and temporally continuous high-accuracy reverence measurements at high resolution.

<div style="border: 4px solid  darkorange; border-radius: 16px; padding: 4px 16px; margin: 1em auto; width: 100%;">
<b>Data Processing in DifERS</b>
<ul>
<li style="padding-bottom: 4px;"> All radiometric images from SSMI/S and ABI are projected onto a common regular $5\times5$ km$^2$ spatial grid;
<li style="padding-bottom: 4px;"> Ground-based precipitation fields from MRMS are remapped onto the same grid and temporally aggregated to hourly resolution;
<li style="padding-bottom: 4px;"> Satellite measurements $\mathbf{z}$ consist of 10 SSMI/S channels acquired at time $t$ and 13 single-band ABI images spanning from $t-30$ to $t+30$ min (<i>i.e.,</i> $\Delta t = 5$&nbsp;min).
</ul>
</div>

At its core, DifERS aims to learn a _probabilistic relationship_ between what satellites observe and what precipitation might actually be occurring on the ground. 

Mathematically, this relationship is written as:
\begin{equation}
  p_{\theta}(\mathbf{x}_0 \mid \mathbf{z})
\end{equation}
which describes the distribution of possible precipitation fields $\mathbf{x}_0$ consistent with the microwave and infrared satellite information contained in $\mathbf{z}$. Here, the parameter set $\theta$ represents the trainable components of the model learned during training.

><mark class="blue"> ___Note__: The training strategy used in DifERS closely follows the standard framework of DDPMs[^Sohl-Dickstein2015]<sup>,</sup>[^Ho2020]. Readers interested in the mathematical and algorithmic details are referred to the original paper[^Guilloteau:2025] and to [Part 1]({{< relref "../Intro-Diffusion-Models-part1/index.md" >}}), which provides an accessible introduction to diffusion models and their training principles._</mark>

Once trained, DifERS can sample from the learned distribution to generate ensembles of realistic precipitation maps wherever and whenever ABI and SSMI/S satellite data are available, providing both an expected estimate and insight into the uncertainty and spatial variability of precipitation.

<center> <span style="letter-spacing: 0.5rem;">• • •</span> </center>

### Ensemble Interpretation

Each one of the 128 precipitation maps produced by DifERS is an equally probable realization of the true precipitation field given satellite measurements and the uncertainties. 
The ensemble mean, _i.e._ the average of all 128 realizations, is the estimate with the lowest magnitude of the errors, on average. The ensemble mean is however always a conservative estimate. 

From the measurements of the ABI and SSMI/S instruments, the likelihood of localized extremes precipitation intensities (exceeding 40 mm/h) to occur can be estimated; the information provided by the instruments is however insufficient to pinpoint the exact location of these extreme intensities at 5 km resolution. 

{{< figure
  align=center
  src="../../images/diff4_DifERS_architeture.png"
  alt="DifERS architecture"
  caption="Schematic representation of the DifERS architecture. <small>Image source: [Guilloteau et al., (2025)](https://ieeexplore.ieee.org/abstract/document/10912662)</small>"
  width=100%
>}}

The extremes intensities often occur at different locations across the 128 DifERS realizations. For this reason, the ensemble mean is generally a spatially smooth field, with dimmed extreme intensities and small variations of the expected precipitation intensity across adjacent pixels. 

The 128 individual realizations, while they are not exact representation of the truth (the high-resolution spatial patterns and the location of the extremes being partially randomized during the generation process), are all realistic precipitation maps, which reproduce the intensity range of the true precipitation maps and the sharp spatial variations of intensity. 

> <mark class="green">Any of the 128 realizations (or all of them) may for example be used as input of an hydrological to simulate realistic runoff and streamflow.</mark> 

{{< figure
  align=center
  src="../../images/diff4_precip_field.png"
  alt="Precipitation fields"
  caption="Precipitation fields generated by DifERS for the period ranging from 12:40 UTC to 13:40 UTC on 2021-05-04 over parts of Illinois and Indiana. Four of the 128 generated realizations are shown on the left, along with the 128-member ensemble mean and ensemble dispersion (standard deviation) and the gauge-radar ground truth. <small>Image source: [Guilloteau et al., (2025)](https://ieeexplore.ieee.org/abstract/document/10912662)</small>"
  width=100%
>}}

<center> <span style="letter-spacing: 0.5rem;">• • •</span> </center>

### Model Evaluation
To evaluate how realistic the DifERS precipitation maps are, they are compared with the gauge-radar derived precipitation maps over the training area. Of particular interest is the statistical distribution of precipitation intensities across multiple scales.

The moments of order 2 to 4 of the distributions (_variance, skewness and kurtosis_) are compared across multiple spatial scales from 5 to 160 km and the DifERS fields are found to match the gauge-radar reference much more closely than satellite-derived estimates from legacy algorithms. 

{{< figure
  align=center
  src="../../images/diff4_extreme.png"
  alt="Histograms"
  caption="Density histograms showing the statistical distribution of precipitation intensities at 5 km and 1-hour resolution in the DifERS fields, along with the distributions for the DifERS ensemble mean (EM) field, the MRMS ground truth and the operational GPROF and PERSIANN-CCS products. <small>Image source: [Guilloteau et al., (2025)](https://ieeexplore.ieee.org/abstract/document/10912662)</small>"
  width=60%
>}}

The location accuracy of precipitation features in the DifERS maps is evaluated through the spatial coherence between the DifERS ensemble mean and the gauge-radar truth, computed at different wavelengths in the Fourier domain. 
In terms of spatial coherence with the truth, DifERS surpasses legacy algorithms at all wavelengths.

This demonstrates the ability of a deep generative diffusion model like DifERS to generate realistic precipitation intensity maps properly constrained by measurements form satellite infrared and microwave imagers.

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## Summary
- Satellite imagers provide frequent but indirect information about precipitation
- Diffusion-based generative models naturally represent uncertainty through ensembles
- DifERS leverages this framework to generate high-resolution, physically realistic precipitation maps constrained by satellite infrared and microwave observations.

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## References

[^Guilloteau:2025]:Guilloteau et al., [A Generative Diffusion Model for Probabilistic Ensembles of Precipitation Maps Conditioned on Multisensor Satellite Observations](https://ieeexplore.ieee.org/abstract/document/10912662), *IEEE Trans. Geosci. Remote Sens.*, **63**, 1--15, (2025).
[^Sohl-Dickstein2015]: Sohl-Dickstein, J. et al., 2015. [Deep unsupervised learning using nonequilibrium thermodynamics](https://arxiv.org/abs/1503.03585). *Proceedings of the 32$^{nd}$ International Conference on Machine Learning (ICML)*, PMLR, 37, pp.2256–2265.
[^Ho2020]: Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*, pp. 6840-6851.
