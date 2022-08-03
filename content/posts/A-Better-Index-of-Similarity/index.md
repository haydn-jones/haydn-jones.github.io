---
title: "A Better Metric of Neural Network Similarity"
date: 2022-06-01T11:50:15-06:00
draft: false
tags: ["ML"]
categories: ["Research"]
author: "Haydn Jones"
summary: "If you train one cat you get a toucan for free."
TocOpen: true
math: true
ShowWordCount: true
cover:
    image: "images/inv_diagram.webp"
    alt: "Image inverion process" # alt text
    caption: "" # display caption under cover
    relative: true # when using page bundles set this to true
    hidden: false # only hide on current single page
---

If you're looking for code to get started with our method, head to [Getting Started With Some Code]({{< relref "#getting-started-with-some-code" >}}).


# A Confound in the Data

[We recently published new results](https://openreview.net/pdf?id=BGfLS_8j5eq) indicating that there is a significant confound that must be controlled for when calculating similarity between neural networks: **correlated, yet distinct features *within* individual data points**. Examples of features like this are pretty straightforward and exist even at a conceptually high level: eyes and ears, wings and cockpits, tables and chairs. Really, any set of features that co-occur in inputs frequently and are correlated with the target class. To see why features like this are a problem let’s first look at the definition of the most widely used similarity metric, [**Linear Centered Kernel Alignment (Kornblith et al., 2019)**](https://arxiv.org/abs/1905.00414), or Linear CKA.


## Linear Centered Kernel Alignment (CKA)
CKA computes a metric of similarity between two neural networks by comparing the neuron activations of each network on provided data points, usually taken from an iid test distribution. The process is simple: pass each data point through both networks and extract the activations at the layers you want to compare and stack these activations up into two matrices (one for each network). We consider the *representation* of an input point to be the *activations* recorded in a neural network at a specific layer of interest when the data point is fed through the network. We compute similarity by mean-centering the matrices along the columns, and computing the following function:

$$
\begin{equation}
    \text{CKA}(A, B) = \frac{
        \lVert cov(A^T, B^T) \rVert_F^2
    }{
        \lVert cov(A^T, A^T) \rVert_F
        \lVert cov(B^T, B^T) \rVert_F
    }
\end{equation}
$$

This will provide you with a score in the range \\([0, 1]\\), with higher values indicating more similar networks. What we see in the equation above, effectively, is that CKA is computing a normalized measure of the covariance between neurons across networks. Likewise, all other existing metrics for network similarity use some form of (potentially nonlinear) feature correlation.

Linear CKA is easily translated into a few lines of PyTorch:

{{< highlight python >}}
import torch
from torch import Tensor

def CKA(A: Tensor, B: Tensor):
    # Mean center each neuron
    A = A - torch.mean(A, dim=0, keepdim=True)
    B = B - torch.mean(B, dim=0, keepdim=True)

    dot_product_similarity = torch.linalg.norm(torch.matmul(A.t(), B)) ** 2

    normalization_x = torch.linalg.norm(torch.matmul(A.t(), A))
    normalization_y = torch.linalg.norm(torch.matmul(B.t(), B))

    return dot_product_similarity / (normalization_x * normalization_y)
{{< / highlight >}}

## Idealized Neurons
With the idea of feature correlation in mind let's picture two networks, each having an idealized neuron. The first network has a {{<color "#F6B819" cat-ear >}} detector neuron--it fires when there are cat ears present in the image and does not otherwise. The other network has a neuron that is quite similar, but this one is a {{<color "#346DB5" cat-tail >}} detector‚ which only fires when cat tails are found. These features are distinct both visually and conceptually, but their neurons will show high correlation: images containing cat tails are very likely to contain cat ears, and conversely when cat ears are not present there are likely to be no cat tails. **CKA will find these networks to be quite similar, despite their reliance on entirely different features.**

## Overcoming the Confound
We need a way to isolate the features in an image used by a network while randomizing or discarding all others (i.e., preserve the {{<color "#346DB5" cat-tail >}} in an image of a cat, while randomizing / discarding every other feature, including the {{<color "#F6B819" cat-ears >}}).

A technique known as **Representation Inversion** can do exactly this. Representation inversion was introduced by [**Ilyas et al. (2019)**](https://arxiv.org/abs/1905.02175) as a way to understand the features learned by robust and non-robust networks. This method constructs model-specific datasets in which all features not used by a classifier are randomized, thus removing co-occurring features that are not utilized by the model being used to produce the inversions.

Given a classification dataset, we randomly choose pairs of inputs that have *different* labels. The first of each pair will be the `seed` image `s` and the second the `target` image `t`. Using the seed image as a starting point, we perform gradient descent to find an image that induces the same activations at the representation layer, \\(\text{Rep}(\cdot)\\), as the target image[^1]. The fact that we are performing a local search is critical here, because there are many possible inverse images that match the activations. We construct this image through gradient descent in input space by optimizing the following objective:

$$
\begin{equation}
    \text{inv} = \min_s \lVert \text{Rep}(s) - \text{Rep}(t) \rVert_2
\end{equation}
$$

By sampling pairs of `seed` and `target` images that have distinct labels we eliminate features correlated with the target class that are not used by the model for classification of the target class. This is illustrated below for our two idealized {{<color "#346DB5" cat-tail >}} and {{<color "#F6B819" cat-ear >}} classifiers:

[^1]: We define the representation layer to be the layer before the fully connected output layer. We are most interested in this layer as it should have the richest representation of the input image.

{{< figure src="images/inv_diagram.webp#center" alt="Image inverion process" >}}

Here, our `seed` image is a toucan and our `target` image is a cat. Representation inversion through our {{<color "#346DB5" cat-tail>}} network will produce an inverted image retaining the pertinent features of the `target` image while ignoring all other irrelevant features, resulting in a toucan with a cat tail. This happens because adding {{<color "#F6B819" cat-ear>}} features into our `seed` image will not move the representation of the `seed` image closer to the `target` as the network utilizes only {{<color "#346DB5" cat-tails>}} for classification of cats.

When our cat tail toucan is fed through both networks, we will find that while our {{<color "#346DB5" cat-tail>}} neuron stays active, our {{<color "#F6B819" cat-ear>}} neuron never fires as this feature isn't present! We’ve successfully isolated the features of the blue network in our target image and now can calculate similarity more accurately. The above figure also illustrates this process for the {{<color "#F6B819" cat-ear>}} network, however, representation inversion under this network produces a toucan with ears rather than a tail.

By repeating this process on many pairs of images sampled from a dataset we can produce a new inverted dataset wherein each image contains only the relevant features for the model while all others have been randomized. **Calculating similarity between our inverting-model and an arbitrary network using the inverted dataset should now give us a much more accurate estimation of their similarity** [^2].

[^2]: It turns out that our metric of network similarity was simultaneously proposed and published by Nanda et al. in ["Measuring Representational Robustness of Neural Networks Through Shared Invariances”]( https://arxiv.org/abs/2206.11939), where they call this metric “STIR”.

# Results
{{< figure src="images/results.webp#center" alt="Image inverion process" >}}

Above we present two heatmaps showing CKA similarity calculated across pairs of 9 architectures trained on ImageNet. The left heatmap shows similarity on a set of images taken from the ImageNet validation set, as is normally done. Across the board we see that similarity is quite high between any pair of architectures, averaging at [TODO]. On the right we calculate similarity between the same set of architectures, however each row and \\(0.67\\) column pair is evaluated using the **row model’s** inverted dataset. Here we see that similarity is actually quite low when we isolate the features used by models!

Alongside these results we investigated how robust training affects the similarity of architectures under our proposed metric and multiple others. **Surprisingly, we found that as the robustness of an architecture increases so too does its similarity to every other architecture, at any level of robustness.** If you’re interested in learning more, we invite you to give the paper a [read](https://openreview.net/pdf?id=BGfLS_8j5eq).

# Getting Started With Some Code
If you’d like to give this method a shot with your own models and datasets, we provide some code to get you started using [PyTorch](https://pytorch.org/) and the [Robustness](https://robustness.readthedocs.io/) library. This code expects your model to be an AttackerModel provided by the Robustness library‚ for custom architectures check the documentation [here](https://robustness.readthedocs.io/en/latest/example_usage/training_lib_part_2.html#training-with-custom-architectures) to see how to convert your model to one, it’s not too hard.

All you need to do is provide the function `invert_images` with a model (AttackerModel), a batched set of seed images, and a batched set of target images (one for each seed image)--all other hyperparameters default to the values used in our paper.

Before you start there are a couple things to double check:
 - Make sure that your seed and target pairs are from different classes.
 - Make sure that your models are in evaluation mode.

{{< highlight python >}}
from typing import Tuple
import torch
from robustness import attack_steps
from robustness.attacker import AttackerModel
from torch import Tensor
from tqdm import tqdm

class L2MomentumStep(attack_steps.AttackerStep):
    """L2 Momentum for faster convergence of inversion process"""

    def __init__(
        self,
        orig_input: Tensor,
        eps: float,
        step_size: float,
        use_grad: bool = True,
        momentum: float = 0.9,
        ):
        super().__init__(orig_input, eps, step_size, use_grad=use_grad)

        self.momentum_g = torch.zeros_like(orig_input)
        self.gamma = momentum

    def project(self, x: Tensor) -> Tensor:
        """Ensures inversion does not go outside of `self.eps` L2 ball"""

        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return torch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x: Tensor, g: Tensor) -> Tensor:
        """Steps along gradient with L2 momentum"""

        g = g / g.norm(dim=(1, 2, 3), p=2, keepdim=True)
        self.momentum_g = self.momentum_g * self.gamma + g * (1.0 - self.gamma)

        return x + self.momentum_g * self.step_size

def inversion_loss(
    model: AttackerModel, inp: Tensor, targ: Tensor
) -> Tuple[Tensor, None]:
    """L2 distance between target representation and current inversion representation"""

    _, rep = model(inp, with_latent=True, fake_relu=False)
    loss = torch.div(torch.norm(rep - targ, dim=1), torch.norm(targ, dim=1))
    return loss, None

def invert_images(
    model: AttackerModel,
    seed_images: Tensor,
    target_images: Tensor,
    batch_size: int = 32,
    step_size: float = 1.0 / 8.0,
    iterations: int = 2_000,
    use_best: bool = True,
) -> Tensor:
    """
    Representation inversion process as described in
    `If You've Trained One You've Trained Them All: Inter-Architecture Similarity Increases With Robustness`

    Default hyperparameters are exactly as used in paper.

    Parameters
    ----------
    `model` : AttackerModel
        Model to invert through, should be a robustness.attacker.AttackerModel
    `seed_images` : Tensor
        Tensor of seed images, [B, C, H, W]
    `target_images` : Tensor
        Tensor of corresponding target images [B, C, H, W]
    `batch_size` : int, optional
        Number of images to invert at once
    `step_size` : float, optional
    'learning rate' of backprop step
    `iterations` : int, optional
        Number of back prop iterations
    `use_best` : bool
        Use best inversion found rather than last

    Returns
    -------
    Tensor
        Resulting inverted images [B, C, H, W]
    """

    # L2 Momentum step
    def constraint(orig_input, eps, step_size):
        return L2MomentumStep(orig_input, eps, step_size)

    # Arguments for inversion
    kwargs = {
        "constraint":  constraint,
        "step_size":   step_size,
        "iterations":  iterations,
        "eps":         1000, # Set to large number as we are not constraining inversion
        "custom_loss": inversion_loss,
        "targeted":    True, # Minimize loss
        "use_best":    use_best,
        "do_tqdm":     False,
    }

    # Batch input
    seed_batches   = seed_images.split(batch_size)
    target_batches = target_images.split(batch_size)

    # Begin inversion process
    inverted = []
    for init_imgs, targ_imgs in tqdm(
        zip(seed_batches, target_batches),
        total=len(seed_batches),
        leave=True,
        desc="Inverting",
    ):
        # Get activations from target images
        (_, rep_targ), _ = model(targ_imgs.cuda(), with_latent=True)

        # Minimize distance from seed representation to target representation
        (_, _), inv = model(
            init_imgs.cuda(), rep_targ, make_adv=True, with_latent=True, **kwargs
        )

        inverted.append(inv.detach().cpu())

    inverted = torch.vstack(inverted)
    return inverted
{{< / highlight >}}

# Conclusions
While I mainly focused on our proposed metric in this article, I briefly wanted to discuss some of the interesting takeaways we included in the paper. The fact that robustness systematically increases a model’s similarity to any arbitrary other model, regardless of architecture or initialization, is a significant one. Within the representation learning community, some researchers have posed a **universality hypothesis**. This hypothesis conjectures that the features networks learn from their data are **universal**‚ in that they are shared across distinct initializations or architectures. Our results imply a *modified* universality hypothesis, suggesting that under sufficient constraints (i.e., a robustness constraint), diverse architectures will converge on a similar set of learned features. This could mean that empirical analysis of a *single* robust neural network can reveal insight into *every* other neural network--possibly bringing us closer to understanding the nature of adversarial robustness itself. This is especially exciting in light of research looking at similarities between the representations learned by neural networks and brains.
<!-- [perhaps cite Teti’s arch-nemesis, [Joel Dapello](https://bcs.mit.edu/directory/joel-dapello) here]. -->

# References
[1] Haydn T. Jones, Jacob M. Springer, Garrett T. Kenyon, and Juston S. Moore. ["If You've Trained One You've Trained Them All: Inter-Architecture Similarity Increases With Robustness."](https://openreview.net/forum?id=BGfLS_8j5eq) UAI (2022).

[2] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey E. Hinton. ["Similarity of Neural Network Representations Revisited."](https://arxiv.org/abs/1905.00414) ICML (2019).

[3] Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Logan Engstrom, Brandon Tran, and Aleksander Madry. [“Adversarial Examples Are Not Bugs, They are Features.”](https://arxiv.org/abs/1905.02175) NeurIPS (2019).

[4] Vedant Nanda, Till Speicher, Camila Kolling, John P. Dickerson, Krishna Gummadi, and Adrian Weller. ["Measuring Representational Robustness of Neural Networks Through Shared Invariances.”]( https://arxiv.org/abs/2206.11939) ICML (2022).


---
LA-UR-22-27916