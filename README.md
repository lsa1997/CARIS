# CARIS: Context-Aware Referring Image Segmentation
This repository is for the ACM MM 2023 paper [CARIS: Context-Aware Referring Image Segmentation](https://dl.acm.org/doi/10.1145/3581783.3612117).

<div align="center">
  <img src="figures/network.png" width="600" />
</div>

## Requirements
The code is verified with Python 3.8 and PyTorch 1.11. Other dependencies are listed in `requirements.txt`.

## Datasets
Please follow the instruction in [.refer](./refer/README.md) to download annotations of RefCOCO/RefCOCO+/RefCOCOg. We provide the combined annotations as refcocom [here](https://drive.google.com/file/d/1_WnCziCIVHXpWYDsIsHbxzH_KCiYhflo/view?usp=sharing).

Download images from [COCO](https://cocodataset.org/#download). Please use the first downloading link *2014 Train images [83K/13GB]*, and extract the downloaded `train_2014.zip` file. 

Data paths should be as follows:
```
.{YOUR_REFER_PATH}
├── refcoco
├── refcoco+
├── refcocog
├── refcocom

.{YOUR_COCO_PATH}
├── train2014
```

## Pretrained Models
Download pretrained [Swin-B](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth) and [BERT-B](https://huggingface.co/bert-base-uncased/tree/main). Check [models](https://huggingface.co/saliu/CARIS) to get pretrained CARIS models.

## Usage
### Train
By default, we use fp16 training for efficiency. To train a model on refcoco with 2 GPUs, modify `YOUR_COCO_PATH`, `YOUR_REFER_PATH`, `YOUR_MODEL_PATH`, and `YOUR_CODE_PATH` in `scripts/train_refcoco.sh` then run:
```
sh scripts/train_refcoco.sh
```
You can change `DATASET` to `refcoco+`/`refcocog`/`refcocom` for training on different datasets. 
Note that for RefCOCOg, there are two splits (umd and google). You should add `--splitBy umd` or `--splitBy google` to specify the split.

### Test
Single-GPU evaluation is supported. To evaluate a model on refcoco, modify the settings in `scripts/test_refcoco.sh` and run:
```
sh scripts/test_refcoco.sh
```
You can change `DATASET` and `SPLIT` to evaludate on different splits of each dataset. 
Note that for RefCOCOg, there are two splits (umd and google). You should add `--splitBy umd` or `--splitBy google` to specify the split. 
For the models trained on `refcocom`, you can directly evaluate them on the splits of `refcoco`/`refcoco+`/`refcocog(umd)`.

## References
This repo is mainly built based on [LAVT](https://github.com/yz93/LAVT-RIS) and [mmdetection](https://github.com/open-mmlab/mmdetection). Thanks for their great work!

## Citation
If you find our code useful, please consider to cite with:
```
@inproceedings{liu2023caris,
  title={CARIS: Context-Aware Referring Image Segmentation},
  author={Liu, Sun-Ao and Zhang, Yiheng and Qiu, Zhaofan and Xie, Hongtao and Zhang, Yongdong and Yao, Ting},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  year={2023}
}
```

