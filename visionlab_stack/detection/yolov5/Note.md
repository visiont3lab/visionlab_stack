
# Teste

> Estimate background from video sequence

* Background subtracto and estimator KNN, MOG2 (Opencv). If people do not move they are marked as background 
* Median filter to estimate background (time consuming). It estimates correctly the background but introduce light problems.
* histogram matching to compensate light problem not working


## Real time inpainting

Image Inpainting

Best:
* [Generative inpainting, support contextual attention and gate](https://github.com/JiahuiYu/generative_inpainting)
    * Tensorflow 1.7, requires neuralgym (Hard to use it)
* [Free Form Image inpainting with Gated Convolution Pytorch](https://github.com/csqiangwen/DeepFillv2_Pytorch)
* [Generative Inpainting Contextual Attention](https://github.com/daa233/generative-inpainting-pytorch)

Interesting:
* [ Image Inpainting With Learnable Bidirectional Attention Maps PyTorch](https://github.com/Vious/LBAM_Pytorch)

Video Inpainting
* [Flow-Edge Guided Video Completion](https://github.com/vt-vl-lab/FGVC)
* [Deep Video Inpainting Pytorch](https://github.com/mcahny/Deep-Video-Inpainting)

Extra:
* [MM Editing (Inpainting, Matting, Super Resolution, Generation) Pytorch](https://github.com/open-mmlab/mmediting)