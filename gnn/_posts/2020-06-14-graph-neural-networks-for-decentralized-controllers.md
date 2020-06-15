---
title: Graph Neural Networks For Decentralized Controllers
---

This paper describes the application of graph convolutional 
neural networks (GCNNs) and graph recurrent neural networks (GRNNs)
to the problem of learning a decentralized controller.
Determination of an optimal decentralized controller is challenging
since an agent's action is determined by its local neighborhood rather
than all agent states. In the GCNN framework a nonlinear map is 
learned from the system state to the control action and 
in the GRNN framework the dynamical evolution of the system is learned. 
In both settings training is performed by immitation learning.
To evaluate the utility of the proposed models, the authors consider
flocking behavior.

### References
Graph Neural Networks For Decentralized Controllers [[pdf]](https://arxiv.org/abs/2003.10280) [[code]](https://github.com/alelab-upenn/graph-neural-networks)
