# Transformer

This is a PyTorch implementation of the Transformer architecture described
in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

The implemented architecture is intended specifically for tackling the Sequence to Sequence
tasks. 

To improve the readability of matrix products inside the Transformer architecture, 
this implementation makes use of the Pytorch 
[Named Tensor API](https://pytorch.org/docs/stable/named_tensor.html) to manipulate 
Tensor dimensions.

