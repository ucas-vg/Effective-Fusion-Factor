### [Installation](#1.)
### [Getting started](#2.)
### [Evaluation](#3.)
### [Experiment](#4.)

---

# Installation <a name='1.'/>

### Requirements:
- PyTorch 1.0.1
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- (optional) OpenCV for the webcam demo


### Step-by-step installation

```bash
# first, make sure that your conda is setuped properly with the right environment
# for that, check `which conda`, `which pip` and `which python`, whether they point to the
# right path. From a new conda env, the following is what you need to do:

conda create --name maskrcnn_benchmark
conda activate maskrcnn_benchmark

# install the right pip and dependencies
conda install ipython

# maskrcnn_benchmark and coco api dependencies
pip install -r requirements.txt

# PyTorch installation

# PyTorch installation
conda install pytorch=1.0.1 torchvision cudatoolkit=10.0 -c pytorch

conda install opencv
conda install scipy

# install pycocotools
cd ${TinyBenchmark}/tiny_benchmark
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
cd ../../

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
# rm build # if needed
python setup.py build develop

# or if you use MacOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```


# Getting started <a name='2.'/>
1. install TinyBenchamrk [Install]()
2. download dataset [dataset](../dataset) and move to \\\${TinyBenchmark}/dataset,
modify path in \\\${TinyBenchmark}/tiny_benchmark/maskrcnn_benchmark/config/paths_catalog.py to your dataset path

```py
class DatasetCatalog(object):
    DATA_DIR = "/home/$user/TinyBenchmark/dataset"
    DATASETS = {
    ....
```
3. download our Scale Match COCO pretrain weight, see [here](../params/Readme.md);<br/>
Or if you want to train your own Scale Match COCO pretrain weight, see [here](configs/TinyCOCO/Readme.md)

4. choose a config file and run as [maskrcnn_benchmark training](https://github.com/facebookresearch/maskrcnn-benchmark#multi-gpu-training)，<font color='ff0000'/>**Attention!!! use "tools/train_test_net.py" instead of "tools/train_net.py"**</font>

perhaps you need to change change MERGE_GT_FILE in config file to gt file path your downlaoded.
```
TEST:
  IMS_PER_BATCH: 2
  COCO_EVALUATE_STANDARD: 'tiny'  # tiny need change
  MERGE_RESULTS: true
  MERGE_GT_FILE: '/home/hui/dataset/tiny_set/annotations/task/tiny_set_test_all.json'  # change to your gt file path
  IGNORE_UNCERTAIN: true
  USE_IOD_FOR_IGNORE: true
```

train and test
```sh
cd ${TinyBenchmark}/tiny_benchmark
export NGPUS=2
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port=9001 tools/train_test_net.py --config ${config_path}
```


# Evaluation <a name='3.'/>

if you use tiny_benchmark to train, the evaluate will run auto. But if you use other code to generate result, you can use script **tiny_benchamrk/MyPackage/tools/evaluate_tiny.py** to evalute, for example
```sh
python evaluate_tiny.py --res ~/results/your_result.json --gt ~/dataset/tiny_set_test_all.json --detail
```
and the mr evaluation is time-cost, you can specified --metric to evaluate AP only
```sh
python evaluate_tiny.py --res ~/results/your_result.json --gt ~/dataset/tiny_set_test_all.json --detail --metric 'ap'
```

Moreover your_result.json should satisfy such format
```
[{'image_id': 793, 'category_id': 1, 'bbox': [0.0, 0.0, 1.0, 1.0], 'score': 0.009999999776482582}, 
{'image_id': 793, 'category_id': 1, 'bbox': [590.0, 0.0, 1.0, 1.0], 'score': 0.009999999776482582},
......
{'image_id': 795, 'category_id': 1, 'bbox': [1180.0, 0.0, 1.0, 1.0], 'score': 0.009999999776482582}]
```


# Experiment <a name='4.'/>

<a color='#00ff00'>  Here are some experimental results. Each group of experiments was run at least 3 times, and the final experimental result was the average of multiple results.</a>

For details of experiment setting, please see [paper](https://arxiv.org/pdf/2011.02298.pdf) Section 4.1. Experiments Setting

training setting| value
---|---
training imageset| dataset/tiny_set/erase_with_uncertain_dataset/train
training annotation| dataset/tiny_set/erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json
test annotation| dataset/tiny_set/annotations/task/tiny_set_test_all.json
deal to ignore region while training| erase with mean color
size of cut image piece| (640, 512)

## 1. detectors

detector | $AP^{tiny}_{50}$ | $AP^{tiny1}_{50}$ | $AP^{tiny2}_{50}$ |  $AP^{tiny3}_{50}$ | $AP^{small}_{50}$| $AP^{tiny}_{25}$| $AP^{tiny}_{75}$
---|---|---|---|---|---|---|---
[FCOS](configs/TinyPerson/fcos/baseline1/fcos_R_50_FPN_1x_baseline1.yaml) 						| 3.26 | 0.99 | 2.82 | 6.2 | 20.19 | 13.28 | 0.14
[RetinaNet](configs/TinyPerson/retina/baseline1/retina_R_50_FPN_1x_baseline1_lr.yaml)			| 33.53 | 12.24 | 38.79 | 47.38  | 48.26 | 61.51 | 2.28
[Adaptive RetinaNet](configs/TinyPerson/retina/baseline1/retina_R_50_FPN_1x_baseline1_lrfpn.yaml)                              |46.56 | 27.08 | 52.63 | 57.88 |  59.97 | 69.6 | 4.49
[Adaptive FreeAnchor](configs/TinyPerson/freeanchor/baseline1/freeanchor_R_50_FPN_1x_baseline1_lrfpn.yaml) | 41.41 | 25.13 | 47.41 | 52.77  | 59.61 | 63.38 | 4.58
[Faster RCNN-FPN](configs/TinyPerson/FPN/baseline1/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_baseline1.yaml)       |47.35| 30.25|51.58|58.95|63.18|68.43|5.83
[Adaptive RetinaNet with S-α](configs/TinyPerson/retina/S_alpha/retina_R_50_FPN_1x_baseline1_lrfpn_sa.yaml)                              | 48.34 | 28.61 | 54.69 | 59.38 | 61.73 | 71.18 | 5.34
[Faster RCNN-FPN with S-α](configs/TinyPerson/FPN/S_alpha/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_baseline1_sa.yaml)                             | 48.39 | 31.68 | 52.20 | 60.01 | 65.15 | 69.32 | 5.78

detector | $MR^{tiny}_{50}$ | $MR^{tiny1}_{50}$ | $MR^{tiny2}_{50}$ |  $MR^{tiny3}_{50}$  | $MR^{small}_{50}$ | $MR^{tiny}_{25}$ | $MR^{tiny}_{75}$
---|---|---|---|---|---|---|---
[FCOS](configs/TinyPerson/fcos/baseline1/fcos_R_50_FPN_1x_baseline1.yaml) 					  | 99.0| 99.96 | 99.77 | 97.68  | 95.49 | 97.24 | 99.89
[RetinaNet](configs/TinyPerson/retina/baseline1/retina_R_50_FPN_1x_baseline1_lr.yaml)			     | 92.66 | 94.52 | 88.24 | 86.52  | 82.84 | 81.95 | 99.13
[Adaptive RetinaNet](configs/TinyPerson/retina/baseline1/retina_R_50_FPN_1x_baseline1_lrfpn.yaml)    | 88.31 |  89.65 | 81.03 | 81.08  | 74.05 | 76.33 | 98.76
[Adaptive FreeAnchor](configs/TinyPerson/freeanchor/baseline1/freeanchor_R_50_FPN_1x_baseline1_lrfpn.yaml) | 89.63 | 88.93 | 80.75 | 83.63  | 74.38 | 78.21 | 98.77
[Faster RCNN-FPN](configs/TinyPerson/FPN/baseline1/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_baseline1.yaml)    |87.57 | 87.86|82.02|78.78|72.56|76.59|98.39 
[Adaptive RetinaNet with S-α](configs/TinyPerson/retina/S_alpha/retina_R_50_FPN_1x_baseline1_lrfpn_sa.yaml)   | 87.73 | 89.51 | 81.11 | 79.49 | 72.82 | 74.85 | 98.57
[Faster RCNN-FPN with S-α](configs/TinyPerson/FPN/S_alpha/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_baseline1_sa.yaml) | 87.29 | 87.69 | 81.76 | 78.57 | 70.75 | 76.58 | 98.42

        

```{.python .input}

```
