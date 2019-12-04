# pix2pix-tensorflow
Separately Layerwise Weights assignment to Generator Network 

## Libraries
1. tensorflow 2.0
2. Numpy
3. termcolor

## Train the pix2pix model
```
CUDA_VISIBLE_DEVICES='0' python pix2pix.py
```

## Test the pix2pix model
Test pix2pix model with assigning layerwise weights to Generator Network obtained from training
```
CUDA_VISIBLE_DEVICES='0' python pix2pix-assigned_weights
```
