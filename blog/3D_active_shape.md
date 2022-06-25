---
layout: research
title: 3D active search
description: Paper on 3D active search
---

# Active 3D Shape Reconstruction
from Vision and Touch

Authors: Edward J. Smith, R. Calandra
Created on: December 14, 2021 11:20 AM
Tags: 3D CV, RL, Touch sensing
URL paper: https://proceedings.neurips.cc/paper/2021/file/8635b5fd6bc675033fb72e8a3ccc10a0-Paper.pdf

# **What**

Humans build 3D understanding of the world via *active exploration.* However, 3D reconstruction has recently relied on *static datasets.* This paper proposes *active touch sensing* by focusing on three steps:

- A haptic simulator for active touching of 3D objects.
- A mesh-based 3D reconstruction model that relies on tactile and visuotactile signals.
- Data driven solutions to drive the shape exploration.

# **Why**

Research is not investigating 3D shape reconstruction using active approaches, but it relies on static datasets such as RGB images, depth maps or haptic readings. Deep learning-based data-driven approaches to active touch for shape understanding are practically non-existent. Haptic exploration works consider objects independently and not not make use of learn priors. This means that a large number of touches are necessary to reconstruct one single object and to guide shape exploration. This paper proposes a new formulation to achieve active touch sensing for 3D reconstruction.

# How

Given a pre-trained shape reconstruction model (pre-trained neural networks) over touch signals and *optionally* vision signals, the goal is to find a sequence of touch inputs that leads to the highest reconstruction accuracy.

.....

# Results

In the case of only touch inputs, the model does not use the shape information, but uses a deterministic trajectory learned from training data. This means that there is not enough information, or the information is hard to extract suing the current methods. In case of touch and vision, the policy tends to favour occluded areas, meaning that the model select graps points based on the current belief of the object shape.

# Limitations

- The reconstruction method tries to minimise the CD (Chamfer Distance), and this leads to poor visual results.
- The hand moves towards the centre of the object, so the touch inputs are biased towards that.
- The environment requires full 3D shape supervision for training, because the simulator requires as input the 3D objects shape. This is possible in a virtual setting, but harder in real-life.

# My comments

- I like it because it combines 3D reconstruction, robotics and reinforcement learning.
- It is the first paper that explores active touch sensing, so there is room for improvements and novelty.
- The paper is relatively easy to understand.
- It seems like RL is not performing very well according to the results. Does this mean that RL is not the best strategy to guide touch inputs?

# References

A few approaches have tried to achieve object 3D reconstruction leveraging touch and vision. (I think) These approaches rely on static datasets and not active touch sensing.