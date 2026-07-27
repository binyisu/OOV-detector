# Zero-Shot Out-of-Vocabulary Object Detection

[[2601.22685] OOVDet: Low-Density Prior Learning for Zero-Shot Out-of-Vocabulary Object Detection, Proceedings of the 43rd International Conference on Machine Learning, 2026](https://icml.cc/virtual/2026/poster/61810)


- ## **Installation**

```bash
git clone https://github.com/binyisu/OOVDet.git
cd OOVDet/

conda create -n oovdet python=3.8 -y
conda activate oovdet

pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

python -m pip install -e detectron2-0.3

pip install -r requirements.txt
```

- ## **Prepare datasets**


You should download：

- train and val set of COCO2017

- trainval and test set of VOC2007、VOC2012

following the structure described below：

```
datasets/
  coco/
  VOC20{07,12}/
```

In coco：

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
```

In  VOC20{07,12}：

```
VOC20{07,12}/
  Annotations/
  ImageSets/
    Main/
      trainval.txt
      test.txt
  JPEGImages/
```

Then we generate all datasets for ZOOV:

```
bash prepare_oov_voc_coco.sh
```

- ## Prepare models

Follow ".\offline_rpn_weights\README.md" and ".\pretrained_ckpt\regionclip\README.md" to prepare pretrained models.

- ## Prepare concepts

Please download the required file from [Google Drive](https://drive.google.com/drive/folders/1b67Lv3GZJdHeaCqa4zFFp92FDhnc2dho?usp=drive_link) and place it in:

```bash
./concepts/
```

- ## Running

  - ##### OOV-COCO dataset settings:

    ```bash
    bash OOV-COCO.sh
    ```

  - ##### OOV-VOC dataset settings:

    ```bash
    bash OOV-VOC.sh
    ```

## Inference / Testing with app.py

You can directly test the trained model using:

```bash
python app.py
```
    
Before running, please download the trained model weights (`model_final.pth`) from [Google Drive](https://drive.google.com/file/d/10y9RQ0Px_KGV70SnG1YBMm-7oPzZRGrL/view?usp=drive_link) and place them in:

```bash
./output/voc/rn50x4/
```


Note that the comm.py, rpn.py, proposal_utils.py and batch_norm.py are modified version based on the [Release v0.3 · facebookresearch/detectron2 (github.com)](https://github.com/facebookresearch/detectron2/releases/tag/v0.3)

@inproceedings{su2026oovdet,
  author    = {Binyi Su and Chenghao Huang and Haiyong Chen},
  title     = {OOVDet: Low-Density Prior Learning for Zero-Shot Out-of-Vocabulary Object Detection},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}

All our experiments were conducted on a single NVIDIA 1080Ti, with a batch size of 1 for base class training and a batch size of 1 for novel class fine-tuning.
