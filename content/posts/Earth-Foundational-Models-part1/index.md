---
title: "Foundation Models for the Earth System: AI That Understands Our Planet"
description: "Foundation models have enabled new AI tools to study Earth system. How do we make the most of this opportunity?"
date: 2025-11-30
tags: ["Foundation Model", "Earth", "Self-supervised learning"]
summary: "Exploring the power of foundation models in AI and Earth sciences."
series: ["Foundation Models"]
author: ["Phong Le"]
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
    # image: "https://www.amd.com/content/dam/amd/en/images/blogs/designs/projects/orbit-2-collaboration/orbit-2-figure-1.png"
    # image: "images/Foundation_Models.png"
    image: "https://media.licdn.com/dms/image/v2/D4E22AQGIYwCHcbBa6Q/feedshare-shrink_2048_1536/feedshare-shrink_2048_1536/0/1728418578513?e=1766016000&v=beta&t=B89Dk6o29uHrU4YwCx7axQgEm1TVz_gEwiWRkrjbyLg"
    # image: "https://cdn.prod.website-files.com/65d8ee5f025f02594c614c17/66ebf5038c1ad0c5af6f7ae3_65f87d7cb00a6e1b7bbb9676_1.webp"
    caption: "Image source: Wildflow"
    relative: false # when using page bundles set this to true
    hidden: false          # don't hide globally
    hiddenInList: true     # hide in list pages
    hiddenInSingle: false  # show inside post
social:
    fediverse_creator: "@Phong Le - ORNL"
editPost:
    URL: "https://github.com/ess-community/aiml-blogs/blob/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

<span style="color: #1E90FF;"><small>[This post is a work in progress -- content will be updated!]</small></span>

Artificial Intelligence (AI) is transforming the way we observe, model, and understand our planet. At the center of this transformation are *foundation models* -- large neural networks trained on huge and diverse datasets.

These models represent a breakthrough beyond deep learning architectures and mark the beginning of a new era in AI: *machines don’t just analyze data, but learn the underlying mechanisms and build a "foundation" for many other applications*.

{{< figure
  src="https://cdn.prod.website-files.com/65d8ee5f025f02594c614c17/66ebf5038c1ad0c5af6f7ae3_65f87d7cb00a6e1b7bbb9676_1.webp"
  alt="Foundation"
  caption="**The evolution of AI models** - from machine learning (feature-centric strategy) to deep learning (model-centric strategy) to foundation models (data-centric strategy). The data-centric approach prioritizes the accumulation of large-scale, high-quality data and, where feasible, aims for end-to-end learning. <small>Image source: [DATAFOREST](https://dataforest.ai/blog/ai-foundation-models-for-big-business-innovation)</small>"
>}}

In this first part of the series, we’ll look at what foundation models are, why they matter, and how they are advancing Earth science research.

<center> <span style="letter-spacing: 1rem;">• • •</span> </center>

## What Are Foundation Models?
If you’ve used tools like ChatGPT, Google Gemini, Midjourney, or GitHub’s Copilot, you’ve already interacted with foundation models.
These are *large AI systems* built at such a scale that they consume a nontrivial portion of the world’s electricity, and we’re at risk of running out of publicly available internet data to train them[^Huyen2024].

Foundation models incorporate complex algorithms and deep learning techniques, allowing them to learn patterns, contexts, and nuances from huge amounts and many types of information (i.e., <mark class="green">*modalities*</mark>) and apply that knowledge to a wide range of tasks.

><mark class="green">A ***modality*** is a distinct source of data that conveys information in a unique way -- e.g., images, audio, text, time series.</mark>

For example, the same AI that helps you draft an email can also:
- Summarize long articles,
- Translate text in another language,
- Generate realistic photos from text prompts.

><mark>***Definition**: Foundation models are neural networks -- trained on extremely large and diverse datasets, generally with self-supervised learning at scale -- that can be adapted to accomplish a broad range of downstream tasks.*</mark>

These remarkable adaptability and versatility are the key features that set foundation models apart from earlier generations of AI.
<!--As AI continues to evolve, foundation models will likely remain pivotal in driving innovations and advancements in AI technologies.-->

<!--So what makes them powerful?-->

<center> <span style="letter-spacing: 0.5rem;">• • •</span> </center>

### One Model, Many Skills
<!--One of the key advantages of foundation models is their role as foundational building blocks for creating specialized downstream applications.-->

For years, AI systems were built like a collection of single-purpose tools.
One translated text. Another recognized faces in photos. Yet another analyzed weather patterns from radar and satellite imagery.
Each was useful -- but anytime a new task came along, we had to build and train a new model from scratch, which is inefficient.

Foundation models flip the script. Instead of learning one task at a time, they learn general skills that transfer to many situations.

Think of learning to cook as an example: instead of memorizing every individual recipe, you learn the basics -- *how to chop ingredients, how flavours work together, how heat changes food, and what spices to use*. Once you understand those fundamentals, you can create new dishes without starting from scratch.

{{< figure
    src="../../images/foundation_model_concept.jpg"
    caption="**Traditional ML vs Foundation models**. Traditional ML models are designed to do one specific task. In contrast, a foundation model can centralize the information from all the data from various modalities. This one model can then be adapted to a wide range of downstream tasks. <small>Source: Adapted from [Armand Ruiz](https://www.linkedin.com/posts/armand-ruiz_how-foundation-models-work-lets-understand-activity-7188522788054282240--8r0/).</small>"
>}}

Foundation models learn in a similar way. By training on broad and diverse datasets, they pick up general patterns and develop flexible skills that can be applied to many different problems.
That’s why we call them *"foundation"* models -- their broad knowledge becomes the base for countless applications.
<!--In other words, foundation models learn the complex distributions across modalities and can use them for solving new tasks.-->

Once <mark class="orange">*pre-trained*</mark>, foundation models can be further <mark class="blue">*fine-tuned*</mark> or even <mark class="purple">*prompted*</mark> to tackle a wide variety of tasks -- including many they were never explicitly trained to perform.

<div style="border: 2px solid #708090; border-radius: 16px; padding: 0px 16px; margin: 1em auto; width: 95%;">
<small>
{{< quote-red >}}
***Pre-train**: learn the basic skills from massive data (initial training phase)* </br>
{{< /quote-red >}}
<div style="margin-top: -18px;"> </div>
{{< quote-blue >}}
***Fine-tune**: customize those skills for a particular task (additional training)*
{{< /quote-blue >}}
<div style="margin-top: -18px;"> </div>
{{< quote-purple >}}
***Prompt**: use natural language to direct a model to perform a new task without retraining it*
{{< /quote-purple >}}
</small>
</div>

Broadly speaking, this flexibility enables AI that grows with our needs -- expanding skills as new challenges arise.

<center> <span style="letter-spacing: 0.25rem;">• • •</span> </center>

### Self-supervised Learning
One of the key innovations behind foundation models possible is *self-supervised learning* (SSL). Instead of relying on expensive human-generated labels, SSL lets models learn directly from raw, unlabeled data by creating their own training signals. This approach gives models wide exposure to real-world information and helps them build a surprisingly deep understanding of of world knowledge.

<!--<div style="border: 2px solid #708090; border-radius: 16px; padding: 0px 16px; margin: 1em auto; width: 95%;">
<small>
{{< quote-red >}}
**Example**: A self-taught chef learns by experimenting with ingredients, cooking techniques, and flavors. They don’t always follow explicit recipes but instead rely on their understanding of how ingredients work together. They taste, adjust, and taste again until they achieve the flavour profile they’re aiming for.
{{< /quote-red >}}
</small>
</div>-->

><mark>***Self-supervised learning**: is a ML approach in which a model learns from unlabeled data by generating its own supervisory signals, often by predicting missing or transformed (e.g., masked, shuffled) parts of the input*.</mark>

We generate overwhelming amounts of unlabeled data every day: text, satellite imagery, sensor signals, and more. Manually labeling it all isn’t just costly -- it’s unrealistic. SSL flips this limitation into an advantage by learning straight from the data itself.

{{< figure
    src="https://lilianweng.github.io/posts/2019-11-10-self-supervised/self-sup-lecun.png"
    alt="Generative"
    caption="An example of how self-supervised learning tasks can be constructed for text data. <small>Image source: LeCun’s talk.</small>"
    width=75%
>}}

In essence, *SSL transforms an unsupervised problem into a supervised one by generating its own “training signals”.* The model sets its own objectives, teasing patterns, structures, and relationships out of massive datasets that would be impossible to label manually.

Because they can learn from huge amounts of raw, uncurated data, foundation models can develop broad, flexible skills. SSL gives them the raw exposure they need to generalize across tasks, making it one of the most important advances behind modern AI.

><mark class="gray">*For a more detailed, mathematical perspective on self-supervised representation learning, see [this post from Lil'Log](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)[^lillog_ssl].*</mark>

<center> <span style="letter-spacing: 0.25rem;">• • •</span> </center>

### Scale and Homogenization
What makes foundation models so powerful is *scale*.
When we train a model on massive, diverse datasets using large amounts of compute, the model doesn’t simply memorize information — it starts to uncover deeper structures and relationships in the data that generalize beyond what it has seen.

{{< quote-red >}}
"If I could use only one word to describe AI post-2020, it’d be *scale*." </br>
-- **Chip Huyen** (AI Engineering)
{{< /quote-red >}}

{{< figure
    src="../../images/FM_pipeline_diagram.webp"
    alt="Generative"
    caption="Pre-training pipeline diagram of foundation models. <small>Image source: [Sebastian Buzdugan](https://medium.com/@sebuzdugan/day-70-100-large-scale-pretraining-and-foundation-models-the-engines-of-modern-ai-19ad552f1305).</small>"
    width=75%
>}}

As models scale up, their abilities tend to improve along familiar metrics. But at sufficiently large scales, we often see something unexpected: new capabilities appear that weren’t directly targeted during training. These <mark class="blue">*emergent behaviors*</mark> can include reasoning in unfamiliar situations, integrating patterns across different modalities, or detecting subtle signals that humans might overlook.

><mark class="blue">***Emergent behaviors:** are characteristics that arise from the interactions inside a large system — not explicitly programmed, and not visible in smaller models.*</mark>

Together, large-scale training and SSL transform models from narrow tools into flexible, general-purpose systems. This leads to an important outcome: <mark class="orange">*homogenization*</mark>.

><mark class="orange">***Homogenization:** The process by which AI systems across diverse tasks and domains converge toward a shared set of large, general-purpose model architectures, training objectives, and data sources, resulting in reduced variation in design and increased reliance on common computational infrastructure.*</mark>

In the past, every scientific task required its own specialized model. If you were predicting wildfires, forecasting ocean temperatures, or estimating crop yields, you’d build everything from scratch -- new architecture, new data pipeline, new rules. Each domain built and maintained its own datasets, pipelines, and assumptions.

Homogenization changes this paradigm. A single foundation model can serve as a shared computational "*backbone*" that many applications and research areas build upon. Instead of numerous disconnected systems, we get a unified base that becomes more capable as more users contribute data, insights, and fine-tuned adaptations.

The result is less fragmentation across the scientific landscape and more opportunities for collaboration. As foundation models continue to scale, they increasingly function as common infrastructure -- a general foundation that supports progress across diverse domains.

Now, these AI models are being applied to one of the most complex systems: **our planet**.

<center> <span style="letter-spacing: 0.75rem;">• • •</span> </center>

## Foundation Models for Earth System
{{< figure
  src="../../images/DOFA1.png"
  alt="Generative"
  caption="**Dynamic One-For-All (DOFA)** - A unified multimodal foundation model for remote sensing and Earth observation. <small>Image source: [Xiong et al., (2024)](https://arxiv.org/pdf/2403.15356)</small>"
>}}

Earth foundation models (EFMs) bring the principles of foundation models from language and vision AI to the study of our planet.
Instead of learning from books, websites, or photographs, EFMs are trained on massive collections of geoscience data: *satellite observations, climate simulations, weather sensor networks, ocean measurements, and more*.

><mark>EFMs help bridge the gap between observation and understanding Earth system -- enabling faster science and more informed actions in a rapidly changing world.</mark>

By learning from this rich and multimodal data, EFMs can develop a holistic view of processes that shape Earth’s systems, such as:
- How weather patterns form
- How oceans and atmosphere interact
- And how environmental changes unfold over time

It is important to note that <mark class="pink">*an EFM typically doesn’t cover the entire Earth’s science all at once*</mark>. Most EFMs are built for a specific domain such as Earth observation, climate modeling, biodiversity, or land use. Still, the knowledge they gain often transfers across applications, giving scientists and users a flexible base to build upon.

<small>

| **Category** | **Scope** | **Training Data** | **Example Models** |
|--------------|-----------|-------------------------|------------------|
| Geospatial | Earth observation | Satellite & aerial imagery, GIS, etc | DOFA[^Xiong2024], Prithvi[^Szwarcman2024], Clay[^clay], TerraMind[^Jakubik2025] |
| Climate / Earth System | Climate modeling, downscaling | reanalysis, ESM outputs, atmospheric data, etc | Earth-2[^Earth2], ClimaX[^Nguyen2023], Aurora[^Bodnar2025], ORBIT-2[^Wang2025] |
| Environment | Environmental monitoring | Fire data, hydrology, soil moisture, precipitation | Granite-Geospatial-Ocean[^ggo] |
| Biodiversity & Ecosystem | Species & habitat monitoring | Camera traps, lidar, bioacoustics, etc | NatureLM-audio[^Robinson2024] |

</small>

{{< quote-purple >}}
<mark>An EFM that covers all major components of the Earth system in a single, unified framework would be considered a **General Earth Foundation Model (GEFM)**$^\dagger$. Achieving this would be a major milestone for the entire field -- but we are not there yet. We will explore the path toward such a model in a future post.</mark> </br>
{{< /quote-purple >}}

<div style="margin-top: -18px;"> </div>
<small>
$^\dagger$This concept reflects the author’s emerging perspective and is shared to encourage discussion rather than imply consensus.
</small>

<div style="margin-top: 24px;"> </div>

Like other foundation models, EFMs are pre-trained on enormous datasets and then fine-tuned for specific applications. This reduces the time, cost, and expertise required to build advanced models from scratch, helping accelerate research, integrate knowledge across domains, and provide a flexible foundation for tackling complex Earth science problems.

### Why Foundation Models for Earth System?

Our planet is changing rapidly, and the pace of those changes demands faster, more integrated science. Weather extremes, shifting ecosystems, rising sea levels -- none of these exist in isolation. They are connected parts of a single, complex system.

However, traditional modeling approaches often treat them separately because each domain requires its own tools, data formats, and expertise.
That separation makes it harder to understand how different parts of the Earth interact, such as:
- How drought influences wildfire behavior
- How ocean warming affects coastal storms
- How land-use change affects regional climates

{{< figure
  src="../../images/why_EFMs.webp"
  alt="Generative"
  caption="EFMs are important for efficiency, generalization, and innovation."
  width=85%
>}}

EFMs help bring these pieces together, ensuring efficiency, generalization, and innovation.
Because they learn from broad and multimodal datasets, EFMs can uncover relationships that span different scientific fields. A model trained on atmospheric data can benefit from what it learns about land or ocean processes. It can generalize to new locations, new time periods, and sometimes new types of data.

 While the initial investment to develop an EFM can be substantial, the long-term benefits are immense. Utilizing pre-trained foundation models significantly speeds up and reduces the cost of developing new machine learning applications compared to training custom models from scratch. This provides a great opportunity:

- Smaller teams can access capabilities that once required massive resources
- Scientists can focus more on discovery and less on building infrastructure
- Early-warning systems can update more frequently with richer information
- Decision-makers can gain clearer insights from a unified view of Earth

Ultimately, EFMs reduce the time between observing a change and understanding what it means -- and that speed matters when lives, ecosystems, and economies are at risk.

### Challenges and Limitations
<!--Data availability and quality
Compute and resource constraints
Integration with existing workflows
Interpretability, market adoption, and trust-->
Building EFMs pushes the boundaries of both AI and geoscience -- and doing so comes with major challenges:

1. **Scale and Infrastructure**: Developing an EFM from scratch requires significant resources, time, and coordination across institutions. This includes expensive hardware, massive amounts of data, and months of training.

2. **Data**:
Earth science data is not evenly distributed. Observations in the Global South, oceans, and polar regions remain sparse, while many datasets lack long historical records or consistent quality. Models inherit these gaps, which can lead to biased predictions in places where insights are most urgently needed.

3. **Cross-Domain Expertise**: EFMs sit at the intersection of advanced AI and Earth system science. Combining these disciplines demands specialized knowledge that remains scarce and unevenly accessible.

4. **Bias, Uncertainty, and Error Propagation**:
Flaws in pre-training can silently scale across every downstream application. Transparent evaluation, benchmarking, and uncertainty quantification are essential but not yet standardized.


**Quick Summary:** </br>
<!--Foundation models aren’t just powering apps -- they’re becoming the digital infrastructure of the future. And we’re only at the beginning of what they’ll enable.-->

- Foundation models represent a major shift in how AI learns — from task-specific tools to flexible, general-purpose systems.
- EFMs apply these advancements to geoscience, learning from massive, multimodal datasets across the Earth system.
- Once pre-trained, EFMs reduce the time and expertise needed to build powerful models for specific applications — accelerating research and decision-making.
- Challenges remain, including compute demands, uneven data coverage, and the need for responsible governance to ensure equitable benefits.

{{< quote-blue >}}
In Part 2, we’ll dive into how foundation models are used in Earth sciences.
{{< /quote-blue >}}

## References:
[^Earth2]: Earth-2. https://www.nvidia.com/en-us/high-performance-computing/earth-2/
[^Bodnar2025]: Bodnar, C., Bruinsma, W.P., Lucic, A. et al. (2025). A foundation model for the Earth system. Nature 641, 1180–1187.
[^Wang2025]: Wang X., et al. (2025). ORBIT-2: Scaling Exascale Vision Foundation Models for Weather and Climate Downscaling. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '25). Association for Computing Machinery, New York, NY, USA, 86–98.
[^Nguyen2023]: Nguyen, T., et al. (2023). ClimaX: a25 foundation model for weather and climate. In Proceedings of the 40th International Conference on Machine Learning (ICML'23), Vol. 202. JMLR.org, Article 1078, 25904–25938.
[^ggo]: Granite-Geospatial-Ocean. https://huggingface.co/ibm-granite/granite-geospatial-ocean
[^Xiong2024]: Xiong, Z., et al., 2024. [Neural plasticity-inspired multimodal foundation model for earth observation](https://arxiv.org/abs/2403.15356). arXiv preprint arXiv:2403.15356.
[^Szwarcman2024]: Szwarcman, D., et al., 2024. [Prithvi-eo-2.0: A versatile multi-temporal foundation model for arth observation applications](https://arxiv.org/abs/2412.02732). arXiv preprint arXiv:2412.02732.
[^Jakubik2025]: Jakubik, J., et al., 2025. Terramind: Large-scale generative multimodality for earth observation. arXiv preprint arXiv:2504.11171.
[^clay]: Clay Foundation Model. https://clay-foundation.github.io/model/
[^Huyen2024]: Huyen, C., 2024. AI Engineering: Building Applications with Foundation Models. O'Reilly Media, Incorporated.
[^lillog_ssl]: https://lilianweng.github.io/posts/2019-11-10-self-supervised/
[^Robinson2024]: Robinson, D., et al., 2024. [NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics](https://arxiv.org/abs/2411.07186). arXiv preprint arXiv:2411.07186.
