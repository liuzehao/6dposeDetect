# 6dposeDetect
A 6D-pose estimation method based on fast-rcnn.
![pic1](https://github.com/liuzehao/6dposeDetect/blob/master/pic1.png)
[![Youtube](https://github.com/liuzehao/6dposeDetect/blob/master/showvideo.png)](https://youtu.be/AyveVaFebcs)
## Test method：
# 1.Download code from github
# 2.Install environment through pip
```
pip install -r requirements.txt
```
# 3.Download trained weights
[GOOGLE DRIVE](https://drive.google.com/drive/folders/1Z7fj3mcl9QljusnHs6kW55vihbuOg9y_?usp=sharing)
put them in:
6dposeDetect/output/res101/voc_2007_trainval/default
# 4.Download data
[GOOGLE DRIVE](https://drive.google.com/drive/folders/1DD5ZOnsbIOcRCn2qnj5MhUkatjOHEij6?usp=sharing)
put them in:
6dposeDetect/data/
# 5.
```python
cd tools
python linemodocclusion9.py
```
You can see the visual results in the folder ./tools/show
![can](https://github.com/liuzehao/6dposeDetect/blob/master/can_00000.jpg)
![cat](https://github.com/liuzehao/6dposeDetect/blob/master/cat_00000.jpg)
![driller](https://github.com/liuzehao/6dposeDetect/blob/master/driller_00000.jpg)
## Train method：
Since there are only 1214 images in line-occulution, we need to use data enhancement to increase the training map. I have one that has been processed. You can also try it yourself according to the project.
# 1.Download data
put them in：
[BAIDU DRIVE](https://pan.baidu.com/s/1izQrfXUbfEG4RwBml1nyVQ )，key:l8j3
6dposeDetect/data/VOCdevkit2007/VOC2007
** The data package is very large. You can decompress it like this：
```
cat test.tar.gz* | tar -xzv
```
# 2.
```python
./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc res101
```
