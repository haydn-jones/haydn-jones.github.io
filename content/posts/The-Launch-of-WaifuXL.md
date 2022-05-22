---
title: "The Launch of WaifuXL"
date: 2022-05-22T14:29:11-06:00
draft: true
tags: []
author: "Haydn Jones"
showToc: true
TocOpen: true
draft: false
hidemeta: false
comments: false
description: ""
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: false
ShowReadingTime: false
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: false
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "/WaifuXL.png" # image path/url
    alt: "WaifuXL Screenshot" # alt text
    caption: "" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false # only hide on current single page
---
# We're Launching!
Today we're finally launching our neural network powered super resolution website, [WaifuXL](https://waifuxl.com/)! This is a project that [The Future Gadgets Lab](https://github.com/TheFutureGadgetsLab/WaifuXL) has been working on for a while and we're really excited to share it with you.

WaifuXL is quite similar to [waifu2x](http://waifu2x.udp.jp/) in function, however, our super resolution model (the [Real-ESRGAN](https://arxiv.org/abs/2107.10833)) produces ***much*** better upsamples, we have a fun image property tagger, and our backend (or lackthereof) is radically different. When you use our service to upscale an image, rather than sending your input to a backend somewhere in the cloud to be upsampled remotely, we send the upsampling neural network (and the tagger) *to you* for execution directly on your laptop, desktop, phone, or tablet. We'll get to how this is possible in a moment, but first we're going to cover the models.

## The Networks
### Super Resolution
What sets the Real-ESRGAN apart from other models (and the models used by waifu2x) is not the architecture but its *training process*. The standard training process for super resolution networks is to simply take a high resolution dataset of images, downscale them to a lower resolution, and train the network to map from the downscaled images to the original high-resolution ones. The Real-ESRGAN training process attempts to directly model the kind of degredations one might encounter in real-world low-quality images through a process they call a "high-order degradation model". During this process they combine many different degradations with various intensities to produce a visually corrupt input to the network. Here is an overview of the various degradations they apply, but refer to **Section 3** of the paper to get a better idea of the entire process:
 - Blur (multiple kinds)
 - Noise
 - Combinations of upsampling and downsampling with various algorithms
 - JPEG compression
 - Sinc filters (to model ringing and overshoot)

### Image Tagging 
For our image tagger we're using a [MobileNetV3](https://arxiv.org/abs/1905.02244). The network will detect 2,000 different characteristics in the image, 2,000 different characters (mostly from anime or related media), and an explicitness rating (safe, questionable, or explicit). Here is an example of some examples of our tagger outputs:
 - Characteristics
   - Number of girls / boys in the image
   - Eye color
   - Hair color
   - Clothing types
 - Characters:
   - Naruto
   - Hatsune Miku
   - Megumin

### The Dataset
To train both of our models we are using the [Danbooru2021](https://www.gwern.net/Danbooru2021) dataset, comprised of ~4.9m images scraped from the Danbooru image board. Due to storage limitations we trained both models on a subset of ~1.2m images, however, this is likely a sizable enough subset to not have caused any problems. The tags in the Danbooru2021 dataset are quite diverse, with over 493,000 distinct tags, but they are *extremely* unbalanced. The most common tag is "1girl" which is present on millions of the images and by the time you get down to the 1,000th most common tag it is only found on ~10,000 images. The unbalanced nature of the tags certainly has resulted in a less than optimal tagger, but there is little that can be done about that.

## The Website
