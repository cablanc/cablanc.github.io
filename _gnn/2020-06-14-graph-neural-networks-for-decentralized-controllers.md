---
title: Graph Neural Networks For Decentralized Controllers
excerpt_separator: <!--more-->
tags: [Graph Neural Networks]
pdf: https://arxiv.org/abs/2003.10280
layout: post
nb: https://mybinder.org/v2/gh/cablanc/graph-neural-networks/be85f4f228c7a1a7427d81a83ef511954f917989
---

This paper describes the application of graph convolutional 
neural networks (GCNNs) and graph recurrent neural networks (GRNNs)
to the problem of learning a decentralized controller.
Determination of an optimal decentralized controller is challenging
<!--more-->
since an agent's action is determined by its local neighborhood rather
than all agent states. In the GCNN framework a nonlinear map is 
learned from the system state to the control action and 
in the GRNN framework the dynamical evolution of the system is learned. 
In both settings training is performed by immitation learning.
To evaluate the utility of the proposed models, the authors consider
flocking behavior.

### References
Graph Neural Networks For Decentralized Controllers [[pdf]](https://arxiv.org/abs/2003.10280) [[code]](https://github.com/alelab-upenn/graph-neural-networks)
