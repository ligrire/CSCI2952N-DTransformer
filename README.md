# CSCI2952N-DTransformer

To train a vision transformer using resnet32 as teacher with all layers, run: 

`python main.py`

To train a vision transformer using resnet56 as teacher with last 3 layers, run:

`python main.py --teacher-model resnet56 --channel-size-lst 64 --div-indices 3 --start-index -3 --name resnet56_last3`


The resnet implementation is borrowed from [here](https://github.com/akamaster/pytorch_resnet_cifar10) with added function to extract feature sequence.

The vision transformer implementation is borrowed from [here](https://github.com/lucidrains/vit-pytorch) with modification on seperation of encoder and classification MLP head.

