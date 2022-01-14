# MirrorGAN
Pytorch implementation of MirrorGAN in the paper [MirrorGAN: Learning Text-to-image Generation by Redescription](https://arxiv.org/abs/1903.05854).
## Prerequisites
* Python 2.7
* Packages
  * pytorch
  * python-dateutil
  * easydict
  * pandas
  * torchfile
  * nltk
  * scikit-image\
* Dataset
  * Download preprocessed metadata for [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) [coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9) and save them to `data/`
  * Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
  * Download [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`
## Usage
* Pre-train DAMSM models
  - For bird dataset: `python code/pretrain_DAMSM.py --cfg cfg/DAMSM/bird.yml`
  - For coco dataset: `python code/pretrain_DAMSM.py --cfg cfg/DAMSM/coco.yml`
* Train MirrorGAN models
  - For bird dataset: `python code/main.py --cfg cfg/bird_attn2.yml`
  - For coco dataset: `python code/main.py --cfg cfg/coco_attn2.yml`
* Use GUI to submit input and show results
  * For machine which open UI: Run `python code/client.py`
  * For machine which save models and run forward-pass: `python code/server.py`
## Reference
* [AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks](https://arxiv.org/abs/1711.10485) ([code](https://github.com/taoxugit/AttnGAN))
