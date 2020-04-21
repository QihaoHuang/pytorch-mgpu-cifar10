# Multi-GPU training with Pytorch.

Let's try easy multi-GPU training with pytorch.

Just adding:

```python
if device == 'cuda':
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True
```
will enable parallel GPU usage in Pytorch! :)

Training dataset: CIFAR10

DeepLearningFramework: Pytorch

Should have more than 2GPUs to test it out.

Should scale more than 2 GPUs!

## Usage

```shell
git@github.com:QihaoHuang/pytorch-mgpu-cifar10.git
cd pytorch-mgpu-cifar10
export CUDA_VISIBLE_DEVICES=0,1 # parallel training with GPUs 0 and 1.

python train_cifar10.py

# parallel training with GPUs 0-3.
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train_cifar10.py

# Res101
python train_cifar10.py --net res101

# Res50
python train_cifar10.py --net res50
```
