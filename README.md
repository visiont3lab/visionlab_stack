# VisionLab Stack

A collection of algorithms to performs computer vision and deep learning tasks.

## Algorithms:

*  **people_remove.py** 
    It allows to remove people from difficult scenario. The objective is to to reach real time performance.
    The pipeline to achive the problem is:
    1. Detection: People [Yolov5 Pytorch](https://github.com/ultralytics/yolov5)
    2. Inpainting: [Deepfillv2 Tensorflow](https://github.com/JiahuiYu/generative_inpainting)
    3. Backgeound Subtractor reduce inpainting temporal error

## Setup

* Python3.6 is required to allow compatibility with [Deepfillv2 Tensorflow](https://github.com/JiahuiYu/generative_inpainting)
    Issue python 3.8 does not support tensorflow 1.14.0

* Cloning the repo:

    ```
    git clone https://github.com/visiont3lab/visionlab_stack.git
    cd visionlab_stack
    mkdir -p data/results/detection
    mkdir -p data/results/images
    mkdir -p data/results/masks
    mkdir -p data/results/inpainting
    mkdir -p data/results/bg_estimate

    virtualenv --python=python3.6 env
    source env/bin/activate
    pip install -r requirements.txt
    ```

* Download models.zip from this [visionlab_stack models link](https://drive.google.com/file/d/1uPLrxxxd1__WVK_xMuDPXGIWojQwBMha/view?usp=sharing). Unzip the file and place it in **data/**

* Run the people removal

    ```
    python people_remove.py
    ```

## Pip Pacakging
    
    ```
    source env/bin/activate
    pip install setuptools wheel 
    # build
    python3 setup.py sdist bdist_wheel
    # install
    pip install dist/visionlab_stack-0.0.1.tar.gz 
    ```

## Results

## TODO

[x] Replace [Deepfillv2 Tensorflow](https://github.com/JiahuiYu/generative_inpainting) with pytorch implementation


## Credits to

* [Yolov5 Pytorch](https://github.com/ultralytics/yolov5)
* [Yolov3/v4 Darknet](https://github.com/AlexeyAB/darknet)
* [Deepfillv2 Tensorflow](https://github.com/JiahuiYu/generative_inpainting)
* [Deepfillv1 Pytorch](https://github.com/vt-vl-lab/FGVC)

