---
title: "Foundation Models for Earth System - An Introduction"
date: 2025-11-24
date: 2025-11-24
tags: ["Foundational Model", "Earth",]
summary: "A beginner’s guide to the powerful models shaping the future of Earth science"
series: ["PaperMod"]
author: ["Phong Le"]
showToc: true
author: ["Phong Le"]
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: true
comments: true
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: false
ShowWordCount: false
ShowRssButtonInSectionTermList: true
UseHugoToc: true
math: true

math: true

cover:
    # image: "https://www.amd.com/content/dam/amd/en/images/blogs/designs/projects/orbit-2-collaboration/orbit-2-figure-1.png"
    image: https://www.olcf.ornl.gov/wp-content/uploads/ORBIT2-OLCF.png
    caption: "Image source: [Oak Ridge National Laboratory - ORNL]"
    relative: false # when using page bundles set this to true
    hidden: false          # don't hide globally
    hiddenInList: true     # hide in list pages
    hiddenInSingle: false  # show inside post
    # image: "https://www.amd.com/content/dam/amd/en/images/blogs/designs/projects/orbit-2-collaboration/orbit-2-figure-1.png"
    image: https://www.olcf.ornl.gov/wp-content/uploads/ORBIT2-OLCF.png
    caption: "Image source: [Oak Ridge National Laboratory - ORNL]"
    relative: false # when using page bundles set this to true
    hidden: false          # don't hide globally
    hiddenInList: true     # hide in list pages
    hiddenInSingle: false  # show inside post
social:
    fediverse_creator: "@Phong Le - ORNL"
    fediverse_creator: "@Phong Le - ORNL"
editPost:
    URL: "https://github.com/ess-community/aiml-blogs/blob/main/content"
    URL: "https://github.com/ess-community/aiml-blogs/blob/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---
AI is revolutionizing the way we learn and understand our planet. One of the most exciting developments in this space for years to come is <mark class="pink">*foundation models for Earth system*</mark>. In this post, we’ll break down what these models are and why they have so much potential.

## What are foundation models?
Imaging you are learning to cook. Instead of only learning how to make one dish — say, spaghetti — you learn:
- How to chop vegetables
- How to boil, bake, and sauté
- What flavors go well together
- How to adjust a recipe when ingredients change
Once you know these general cooking skills, you can make hundreds of meals without starting from zero each time.

A foundation model works the same way: It learns broad skills from a huge amount of data — and then we can ask it to do many different tasks by applying those skills in a new context.

<mark>*Foundation models are large-scale AI systems, trained on extremely broad and diverse datasets (often in a self-supervised manner), that can be adapted or fine-tuned for numerous downstream tasks.*</mark>

They are a turning point in AI, replacing traditional models that were trained for a single purpose. Once pre-trained, foundation models can be adapted to numerous downstream applications with little to no additional training samples.

The term “foundation model” comes from the idea that these models serve as a foundation for many applications. They’re not built for just one task — they’re designed to be adapted and fine-tuned for a variety of uses.
Let’s break down how they work:

{{< figure
  src="https://miro.medium.com/v2/resize:fit:2000/format:webp/1*WbR5Hcz5pAavVCZ3bkhjlA.jpeg"
  alt="Generative"
  caption="Three stages of how foundation models work ([Source](https://x.com/iscienceluvr/status/1592860024657051649))."
>}}


{{< figure
  src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*mvqjTg_K_tsCdrN7bEIGlw.png"
  alt="Generative"
  caption="https://medium.com/@nikitaparate9/revolutionising-ai-with-foundation-models-3332f693f790"
>}}

## What are foundation models for Earth system?
Earth foundation models apply this paradigm to geoscience, climate dynamics, and environmental monitoring.
They are trained on petabytes of global Earth data — from satellites, weather sensors, climate simulations, and geospatial archives — allowing them to understand how the planet behaves across space and time. These models can:

- Enhance weather and climate prediction
- Detect and forecast natural hazards (wildfires, floods, severe storms)
- Monitor land-use change, ecosystems, ice sheets, and oceans
- Support real-time environmental decision-making
- Enable high-resolution climate projections at low computational cost

{{< figure
  src="../../images/DOFA1.png"
  alt="Generative"
  caption="A generative model learns features from the training data and can generate new, similar, and high-quality contents ([Source](https://x.com/iscienceluvr/status/1592860024657051649))."
>}}

They bring five key characteristics that make them revolutionary:
1. Pretrained (using large data and massive compute so that it is ready to be used without any additional training)
2. Generalized — one model for many tasks (unlike traditional AI which was specific for a task such as image recognition)
3. Adaptable (through prompting — the input to the model using say text)
4. Large (in terms of model size and data size e.g. GPT-3 has 175B parameters and was trained on about 500,000 million words, equivalent of over 10 lifetimes of humans reading nonstop!)
5. Self-supervised (see footnote 1) — no specific labels are provided and the model has to learn from the patterns in the data which is provided — see the cake illustration below.

> Earth foundation models help bridge the gap between observation and understanding — enabling faster science and more informed actions in a rapidly changing world.

{{< figure
  src="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41586-025-09005-y/MediaObjects/41586_2025_9005_Fig1_HTML.png?as=webp"
  alt="Generative"
  caption="A generative model learns features from the training data and can generate new, similar, and high-quality contents ([Source](https://x.com/iscienceluvr/status/1592860024657051649))."
>}}
