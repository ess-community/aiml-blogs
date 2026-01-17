---
title: "Diffusion Models (Part 4): Satellite Precipitation Estimation"
description: "Diffusion models are transforming how we analyze and predict Earth system processes"
summary: "Diffusion models for environmental science"
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
    caption: "Image source: [Geostationary Operational Environmental Satellites - R Series](https://gpm.nasa.gov/missions/GPM)."
    relative: true # when using page bundles set this to true
    hidden: false          # don't hide globally
    hiddenInList: true     # hide in list pages
    hiddenInSingle: false  # show inside post
editPost:
    URL: "https://github.com/ess-aiml/blogs/blob/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

This is part 4 of the diffusion model series. In this post, we'll look at how diffusion models are being used for precipitation monitoring and mapping from satellite.

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## Precipitation monitoring and mapping from satellite
Precipitation (e.g., rain, snow, and hail) plays a central role in Earth’s water cycle. Accurate precipitation monitoring and mapping help with everything from daily weather forecasts to flood warnings, water resource management, and long-term climate studies.

One of the most reliable ways to measure precipitation is with <mark class="green">weather radars</mark>.
They can detect and quantify falling rain, snow, and hail while they’re still in the air. 
Most ground-based weather radars use the Doppler effect to measure the velocity of precipitation particles.
<!--Radars can operate from the ground, but also from airplanes or satellites, allowing for extended spatial coverage.-->

{{< figure
  align=center
  src="https://static.scientificamerican.com/dam/m/659d364a7431b48f/original/doppler-radar_graphic_d1.png?m=1748551974.773&w=2000"
  alt="gencast model"
  caption="Weather radars are remote sensing instruments that transmit electromagnetic pulses into the atmosphere and analyze the returned signals after they scatter from precipitation. <small>Image source: [Amanda Montanez](https://www.scientificamerican.com/article/how-doppler-radar-lets-meteorologists-predict-weather-and-save-lives/)</small>"
  width=80%
>}}

{{< figure
  align=center
  src="https://www.goes-r.gov/imagesContent/spacecraft/labelledImageMaps/spacecraftLabelled-front_right.png"
  alt="gencast model"
  caption="Weather radars are remote sensing instruments that transmit electromagnetic pulses into the atmosphere and analyze the returned signals after they scatter from precipitation. <small>Image source: [Amanda Montanez](https://www.scientificamerican.com/article/how-doppler-radar-lets-meteorologists-predict-weather-and-save-lives/)</small>"
  width=70%
>}}


Radars can also operate from airplanes or satellites. And we will focus on satellite-borne radars in this post.
Yet weather radars come with some limitations and operational constraints. 
- Weather radars have a range of a few hundred kilometers at most. Therefore, radars onboard satellites must operate from low orbits, limiting their fields of view. 
- Radars are heavy instruments and require a lot of power, only the largest satellite platforms can carry them, and putting them into orbits is a costly operation. 

Because of this, only a handful of weather radars have been launched into orbit. These few radars are not enough to monitor precipitation around the globe at every hour of the day. 

Fortunately, there exist other instruments that can measure electromagnetic waves coming from the atmosphere and provide information about clouds and their ice and liquid water content. These instruments, called imagers, measure the radiances coming from the clouds at different wavelength in the infrared and microwave domain. While the revisit time of satellite weather radars is about a few days on average at any location on the globe, the revisit time for microwave imagers, when taking all the existing ones together, is about a few hours. For infrared imagers the revisit time is between 5 and 30 minutes.

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## From cloud’s spectral signatures to precipitation intensity
 The combination of the radiances measured by imagers in orbit within a given time frame forms the electromagnetic spectral signature of the clouds. From a cloud spectral signature, with expert knowledge, one can infer the likelihood of the cloud to produce rain, snow or hail, at various intensity levels. 
 
 In recent years, deep-learning algorithms have proven very efficient at the task of estimating the most likely precipitation type and intensity for a given observed spectral signature [ref.]. Such retrieval algorithms are trained using radar-derived precipitations fields where they are available as the retrieval target, collocated with the radiance measurements form the imagers in orbit. The information provided by the spectral signature is however only partial, and some degree of uncertainty always exist regarding the precipitation intensity associated to a given cloud at a given time. 
 
 To handle this uncertainty, the deep-learning retrieval algorithms may be probabilistic, and provide the probability of occurrence of precipitation of different types and at different intensity levels associated to any given spectral signature. 
 
 Alternatively, conditional deep-learning generative algorithms can produce large ensembles of possible precipitation type and intensity fields for a given set of spectral signatures measured within a certain area and a certain time frame. Generative algorithms can produce tens or even hundreds of realizations, covering all possible solutions.

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## The DifERS algorithm for generating probabilistic ensembles of precipitation fields from spectral signatures
The DifERS algorithm (Diffusion-Based Ensemble Rainfall estimation from Satellite) utilizes a diffusion model to produce precipitation intensity maps from combined measurements from the Advanced Baseline Imager (ABI) onboard the GOES-R satellite series and the Special Sensor Microwave Imager / Sounder (SSMI/S) onboard the DMSP satellite series. The DifERS algorithm takes all available SSMI/S and ABI measurement within a one-hour time frame over a 320 km by 320 km area as inputs, and produces an ensemble of 128 possible maps the of hourly-averaged precipitation intensity at a resolution of 5 km. The target precipitation intensity fields used during the training of DifERS are derived from ground radar and gauge measurements over the US, in areas where the radar and gauge coverage allows for spatially and temporally continuous high-accuracy reverence measurements at high resolution. Once the algorithm is trained precipitation maps can be generated wherever and whenever the ABI and SSMI/S data is available.

Each one of the 128 precipitation maps produced by DifERS is an equally probable realization of the true precipitation field given the SSMI/S and ABI measurements and the uncertainties. The ensemble mean, i.e. the average of all 128 realizations, is the estimate with the lowest magnitude of the errors, on average. The ensemble mean is however always a conservative estimate. From the measurements of the ABI and SSMI/S instruments the likelihood of localized extremes precipitation intensities (exceeding 40 mm/h) to occur can be estimated; the information provided by the instruments is however insufficient to pinpoint the exact location of these extreme intensities at 5 km resolution. The extremes intensities often occur at different locations across the 128 DifERS realizations. For this reason, the ensemble mean is generally a spatially smooth field, with dimmed extreme intensities and small variations of the expected precipitation intensity across adjacent pixels. The 128 individual realizations, while they are not exact representation of the truth (the high-resolution spatial patterns and the location of the extremes being partially randomized during the generation process), are all realistic precipitation maps, which reproduce the intensity range of the true precipitation maps and the sharp spatial variations of intensity. Any of the 128 realizations (or all of them) may for example be used as input of an hydrological to simulate realistic runoff and streamflow.

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## Evaluation of DifERS precipitation intensity maps’ accuracy and realism
To evaluate how realistic the DifERS precipitation maps are, they compared are with the gauge-radar derived precipitation maps over the training area. Of particular interest is the statistical distribution of precipitation intensities across multiple scales. The moments of order 2 to 4 of the distributions (variance, skewness and kurtosis) are compared across multiple spatial scales from 5 to 160 km and the DifERS fields are found to match the gauge-radar reference much more closely than satellite-derived estimates from legacy algorithms. The location accuracy of precipitation features in the DifERS maps is evaluated through the spatial coherence between the DifERS ensemble mean and the gauge-radar truth, computed at different wavelengths in the Fourier domain. Again, in terms of spatial coherence with the truth, DifERS surpasses legacy algorithms at all wavelengths.

This demonstrates the ability of a deep generative diffusion model like DiFERS to generate realistic precipitation intensity maps properly constrained by measurements form satellite infrared and microwave imagers.
