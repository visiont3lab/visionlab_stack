# Fast Detectors to apply to Real Time Scenario 

This is a package aiming at collect detectors that can be use in realt time sceneario such as survelliance.

* Video Capture 
* License Plate Detector
* People Detector
* Remove Element from video

## Build Notes

```
cd fast_detectors
virtualenv --python=python3.8 env
source env/bin/activate
python -m pip install --upgrade build
python -m build
pip install dist/fast_detectors-0.0.1-py3-none-any.whl
```

## Yolov5 setup

```
cd ai_library
git clone https://github.com/ultralytics/yolov5.git
# Downlaod requirements pycocotools issue 
pip install -r requirements-modified.txt  
# download weights
chmod a+x weights/download_weights.sh && ./weights/download_weights.sh
# Run
python detect.py --source ../storage/videos/rome-640-480.mp4 --device cpu --view-img --conf 0.1 --weights weights/yolov5s.pt
python detect.py --source ../storage/videos/rome-1920-1080-10fps.mp4 --device 0 --view-img --conf 0.1 --weights weights/yolov5s.pt --img-size 1080 --class 0
python detect.py --source /home/manuel/visiont3lab-github/public/people-remove/images/input/Castenaso --device cpu --view-img --conf 0.1 --weights weights/yolov5m.pt --img-size 1080  --class 0
python detect.py --source /home/manuel/visiont3lab-github/public/people-remove/images/input/Molino-Ariani --device cpu --view-img --conf 0.1 --weights weights/yolov5m.pt --img-size 1080 --class 0
python detect.py --source /home/manuel/visiont3lab-github/public/people-remove/images/input/CMB --device cpu --view-img --conf 0.1 --weights weights/yolov5m.pt --img-size 1980  --class 0
python detect.py --source /home/manuel/visiont3lab-github/public/people-remove/images/input/Castenaso --device cpu --view-img --conf 0.1 --weights weights/yolov5m.pt --img-size 1080 --class 0

# Resample video
ffmpeg -i ../storage/videos/rome-1920-1080.mp4 -filter:v fps=10 ../storage/videos/rome-1920-1080-10fps.mp4
ffmpeg -i ../storage/videos/tokyo-1920-1080.mp4 -filter:v fps=10 ../storage/videos/tokyo-1920-1080-10fps.mp4

```

## Install python3.8 Ubuntu 18.04 LTS

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.8 python3-distutils -y
```

# References

* [Packaging python](https://packaging.python.org/tutorials/packaging-projects/)
* [Yolov5 Pytorch](https://github.com/ultralytics/yolov5)
* [Yolo Darknet Darknet](https://github.com/AlexeyAB/darknet)