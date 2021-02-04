# Noise Optimization For Aritificial Neural Network   
Code for the implementation of "Noise Optimization For Neural Network"  
## Abstract  
Adding noises to artificial neural network(ANN) has been shown to be able to  improve robustness in previous work.   
In this work, we propose a new technique to compute the pathwise stochastic gradient estimate with respect to the standard deviation of the Gaussian noise added to each neuron of the ANN.   
By our  proposed technique, the gradient estimate with respect to noise levels is a byproduct of the backpropagation algorithm for estimating gradient with respect to synaptic weights in ANN.  
Thus, the noise level for each neuron can be optimized simultaneously in the processing of training the synaptic weights at nearly no extra computational cost.   
In numerical experiments, our proposed method can achieve significant performance improvement on robustness of several popular ANN structures under both black box and white box attacks tested in various computer vision datasets.    
## Running Environment  
python 3.7   
pytorch 1.6.0  
## Usage Description   
the model files define the architechture of the evalated models including MLP and  CNN for MNIST, ResNet18 for Cifar-10 and ResNet34 for Tiny-ImageNet.   
All of the models have three implementations respectively for not adding noise,  adding non optimized noise and adding optimized noise.  
## Experiment    
#### On MNIST  
MLP: not adding noise  
MLP+: adding non optimized noise  
MLPN: adding optimized noise  

| Models | Activations |  Ori  | White box robustness   evaluation |        |       | Black box robustness   evaluation |         |            |          |       |        |  
|:------:|:-----------:|:-----:|:---------------------------------:|:------:|:-----:|:---------------------------------:|:-------:|:----------:|:--------:|:-----:|:------:|  
|        |             |       | FGSM                              | L-BFGS | PGD   | Gaussian                          | Impluse | Glass Blur | Contrast | FGSM  | L-BFGS |  
|   MLP  | ReLU        | 0.948 | 0.149                             | 0.255  | 0.105 | 0.935                             | 0.934   | 0.879      | 0.598    | 0.314 | 0.69   |  
|        | Sigmoid     | 0.936 | 0.280                             | 0.324  | 0.207 | 0.883                             | 0.783   | 0.885      | 0.676    | 0.410  | 0.749 |  
|  MLP+  | ReLU        | 0.884 | 0.283                             | 0.413  | 0.238 | 0.875                             | 0.851   | 0.531      | 0.531    | 0.34  | 0.742  |  
|        | Sigmoid     | 0.875 | 0.314                             | 0.433  | 0.253 | 0.869                             | 0.834   | 0.817      | 0.605    | 0.432 | 0.736  |  
|  MLPN  | ReLU        | 0.921 | 0.295                             | 0.42   | 0.267 | 0.895                             | 0.909   | 0.835      | 0.672    | 0.43  | 0.745  |  
|        | Sigmoid     | 0.957 | 0.336                             | 0.568  | 0.275 | 0.946                             | 0.944   | 0.92       | 0.71     | 0.465 | 0.788  |  

CNN: not adding noise  
CNN-MLP+: adding non optimized noise on MLP   
CNN-A+: adding non optimized noise both on Conv and  MLP  
CNN-MLPN: adding non optimized noise on MLP    
CNN-AN: adding non optimized noise both on Conv and  MLP  

|  Models  |  Ori  | White box  evaluation |        |       | Black box evaluation |         |            |          |       |        |  
|:--------:|:-----:|:---------------------:|:------:|:-----:|:--------------------:|:-------:|:----------:|:--------:|:-----:|:------:|  
|          |       | FGSM                  | L-BFGS | PGD   | Gaussian             | Impluse | Glass Blur | Contrast | FGSM  | L-BFGS |  
| CNN      | 0.986 | 0.744                 | 0.616  | 0.655 | 0.983                | 0.971   | 0.752      | 0.845    | 0.917 | 0.779  |  
| CNN-MLP+ | 0.98  | 0.788                 | 0.613  | 0.684 | 0.977                | 0.955   | 0.564      | 0.794    | 0.924 | 0.767  |  
| CNN-A+   | 0.974 | 0.757                 | 0.586  | 0.704 | 0.951                | 0.947   | 0.835      | 0.575    | 0.92  | 0.775  |  
| CNN-MLPN | 0.99  | 0.87                  | 0.685  | 0.752 | 0.995                | 0.984   | 0.788      | 0.853    | 0.957 | 0.818  |  
| CNN-AN   | 0.982 | 0.783                 | 0.766  | 0.714 | 0.976                | 0.973   | 0.867      | 0.834    | 0.928 | 0.826  |  

#### On Cifar-10     
ResNet18: not adding noise   
ResNet18-MLP+: adding non optimized noise on MLP    
ResNet18-A+: adding non optimized noise both on Conv and  MLP   
ResNet18-MLPN: adding non optimized noise on MLP     
ResNet18-AN: adding non optimized noise both on Conv and  MLP  


|     Models     |  Ori  | White box robustness   evaluation |        |       | Black box robustness   evaluation |         |            |          |       |        |  
|:--------------:|:-----:|:---------------------------------:|:------:|:-----:|:---------------------------------:|:-------:|:----------:|:--------:|:-----:|:------:|  
|                |       | FGSM                              | L-BFGS | PGD   | Gaussian                          | Impluse | Glass Blur | Contrast | FGSM  | L-BFGS |  
| ResNet18       | 0.902 | 0.234                             | 0.433  | 0.114 | 0.558                             | 0.53    | 0.189      | 0.544    | 0.467 | 0.562  |  
| ResNet18-MLPN+ | 0.874 | 0.254                             | 0.461  | 0.118 | 0.553                             | 0.535   | 0.185      | 0.536    | 0.469 | 0.554  |  
| ResNet18-A+    | 0.877 | 0.219                             | 0.401  | 0.143 | 0.572                             | 0.543   | 0.184      | 0.544    | 0.493 | 0.570   |  
| ResNet18-MLPN  | 0.899 | 0.368                             | 0.45   | 0.181 | 0.553                             | 0.514   | 0.175      | 0.533    | 0.482 | 0.584  |  
| ResNet18-AN    | 0.905 | 0.393                             | 0.489  | 0.203 | 0.587                             | 0.557   | 0.175      | 0.559    | 0.562 | 0.613  |  


#### On Tiny-ImageNet     
ResNet34: not adding noise   
ResNet34-MLP+: adding non optimized noise on MLP    
ResNet34-A+: adding non optimized noise both on Conv and  MLP   
ResNet34-MLPN: adding non optimized noise on MLP     
ResNet34-AN: adding non optimized noise both on Conv and  MLP  

|     Models     |  Ori  | White box robustness   evaluation |        |       | Black box robustness   evaluation |         |            |          |       |        |  
|:--------------:|:-----:|:---------------------------------:|:------:|:-----:|:---------------------------------:|:-------:|:----------:|:--------:|:-----:|:------:|  
|                |       | FGSM                              | L-BFGS | PGD   | Gaussian                          | Impluse | Glass Blur | Contrast | FGSM  | L-BFGS |  
| ResNet34       | 0.436 | 0.082                             | 0.321  | 0.019 | 0.397                             | 0.351   | 0.341      | 0.331    | 0.374 | 0.329  |  
| ResNet34-MLPN+ | 0.434 | 0.076                             | 0.324  | 0.022 | 0.383                             | 0.339   | 0.323      | 0.333    | 0.362 | 0.312  |  
| ResNet34-A+    | 0.177 | 0.012                             | 0.145  | 0.011 | 0.165                             | 0.155   | 0.138      | 0.133    | 0.158 | 0.145  |  
| ResNet34-MLPN  | 0.445 | 0.119                             | 0.402  | 0.051 | 0.406                             | 0.364   | 0.336      | 0.339    | 0.389 | 0.344  |  
| ResNet34-AN    | 0.448 | 0.121                             | 0.402  | 0.055 | 0.412                             | 0.375   | 0.352      | 0.346    | 0.389 | 0.35   |  




