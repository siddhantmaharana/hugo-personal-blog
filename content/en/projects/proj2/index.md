---
title: "Counting Objects"
weight: 3
---

Counting the products on the image seems to be a challenge in modern retail business. This project used various open CV techniques and various state-of-the-art deep learning algorithms using Object Detection Tensorflow API.

The OpenCV techniques involved thresholding the images, detecting edges, finding contours and then counting the contours in order to get the count of the objects in the image.

Tensorflow Object detection API was also used on pre-trained COCO dataset to detect and count the number of images in the picture. For this project, SSD model with MobileNet and RCNN model was tried on the images, with the latter one outperforming all other techniques.

[Link to Code](https://github.com/siddhantmaharana/counting-objects)
