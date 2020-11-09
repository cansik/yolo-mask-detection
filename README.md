# YOLO Mask Detection

Fast scene mask detection for real world images based on the [darknet](https://pjreddie.com/darknet/yolo/) framework and it's [YOLOv3 Tiny PRN architecture](https://github.com/WongKinYiu/PartialResidualNetworks). 

Now also **YOLOv4** and **YOLOv4-tiny** are supported!

![Demo](readme/bag.jpg)

*Image Source: [Twitter @BAG_OFSP_UFSP](https://twitter.com/BAG_OFSP_UFSP/status/1321126674692530176)*

### Dataset
The dataset for this pre-trained network is provided by [VictorLin000](https://github.com/VictorLin000/YOLOv3_mask_detect) and contains 678 images of people with and without masks. In total there are 3 different classes annotated:

* `NONE` - No mask at all.
* `BAD` - Partial covered face.
* `GOOD` - Mask coveres the essential parts.

You can download the dataset directly from [google drive](https://drive.google.com/drive/folders/1aAXDTl5kMPKAHE08WKGP2PifIdc21-ZG).

### Training
The model has been trained on a 1080TI for about 2h over 6000 iteration with a batch size of 64 and 16 subdivisons (PRN & yolov4-tiny), 64 / 64 for yolov4.

![Char](readme/multi-chart.png)

*YOLOv3-tiny-prn / YOLOv4 / YOLOv4-tiny*

To train the network yourself download the [dataset](https://drive.google.com/drive/folders/1aAXDTl5kMPKAHE08WKGP2PifIdc21-ZG), extract it into the `training` folder (`/training/yolo/Mask_1.jpg`) and download the [initial weights](https://drive.google.com/file/d/18v36esoXCh-PsOKwyP2GWrpYDptDY8Zf/view) into the `training` folder. To start the training run the following command:

```bash
darknet detector train obj.data yolov3-tiny-prn.cfg yolov3-tiny.conv.11
```

After the training, the resulting weights should be in the `/training/backup/` folder.
For instructions regarding YOLOv4, head over to [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet).

### Inferencing
The weights have been trained on an image size 416x416 (PRN & tiny) / 608x608 (YOLOv4). It is also possible to inference it with a lower model size to increase the speed. A good performance / accuracy mix on CPUs has been discovered by using an image size of 128x128.

The model itself is fully compatible with the opencv dnn module and just ready to use.

### Demo
There are either a python or a processing (java) demo you can run to test out your model directly with a webcam.

#### Python
To run the demo, please first install all the dependencies (`requirements.txt`) into a virtual environment and download the model and weights into the model folder (or run the shell script).

Then run the following command to start a webcam detector with Yolov3-Tiny-PRN:

```python
# with python 3
python demo_webcam.py -n prn

# yolov4
python demo_webcam.py -s 608

# yolov4-tiny
python demo_webcam.-n tiny
```

#### Processing
For artists and makers there is also a [basic example](https://github.com/cansik/yolo-mask-detection/blob/main/MaskP5/MaskP5.pde) on how to use this network in [Processing](https://processing.org/). The sketch itself is based on the following libraries:

* [Video Library 2.0](https://github.com/processing/processing-video)
* [Deep Vision Library](https://github.com/cansik/deep-vision-processing)

### Download
To download the models just run the `download-models.sh` in `/models` or use the following links:

* [mask-yolov3-tiny-prn.cfg](https://github.com/cansik/yolo-mask-detection/releases/download/pre-trained/mask-yolov3-tiny-prn.cfg)
* [mask-yolov3-tiny-prn.weights](https://github.com/cansik/yolo-mask-detection/releases/download/pre-trained/mask-yolov3-tiny-prn.weights)

If you are interested into the full size YOLOv3 pre-trained network, head over to this repository: [VictorLin000/YOLOv3_mask_detect](https://github.com/VictorLin000/YOLOv3_mask_detect)

### About

Trained by [cansik](https://github.com/cansik), the dataset fall under the terms and conditions of their owners ([VictorLin000](https://github.com/VictorLin000)).

Demo images are from [Unsplash](https://unsplash.com/):

* [Jérémy Stenuit](https://unsplash.com/photos/3mErKfgolzM)
* [Victor He](https://unsplash.com/photos/lf1ivjbtF2Q)