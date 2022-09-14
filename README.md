# Neural-network-triggers-for-LHC_CMS
University of Bristol Summer Research

Title:
Using Machine Learning to develop fast selection algorithms for the CMS detector at the LHC

The LHC can generate hundreds of millions of data in microseconds. But physicists are interested in only some of them. Our research is based on the LHC level 1 layer. Aims to filter out data of interest using neural networks. At present, the signal data collected by the LHC collector can be stored in 2D image format, so we can apply convolutional neural network for learning. The goal of the study was to find a new type neural networks to filter signals.

-----------------------

We proposed a new machine learning method for particles signals selection. The core of the method is to treat the collision events as a particle cloud, a disordered set of particles. And we improved the data processing method to be more explainable and natural. Based on the particle cloud representation, we introduced the ParticleNetLite++ architecture with special edegconv operation, a dynamic graph convolutional neural network architecture. The experimental results indicated that the model’s accuracy and efficiency can reach 0.9997 at hundred-level parameter size which improved a lot compared to CNN.
