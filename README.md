# WSI-FT (Whole Slide Image Finetuning )
This is the official repository for our CVPR 2023 paper.

[Task-Specific Fine-Tuning via Variational Information Bottleneck for Weakly-Supervised Pathology Whole Slide Image Classification](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Task-Specific_Fine-Tuning_via_Variational_Information_Bottleneck_for_Weakly-Supervised_Pathology_Whole_CVPR_2023_paper.html)

## Stage-1a (baseline): 
Mostly folked from [CLAM](https://github.com/mahmoodlab/CLAM), with minor modification. So just use follow the [docs](https://github.com/mahmoodlab/CLAM/tree/master/docs) to perform baseline, or with following steps:

1) Preparing grid patches of WSI without overlap.
```
bash create_patches.sh
```
2) Preparing pretrained patch features fow WSI head training.
```
bash create_feature.sh
```
3) Training baseline WSI model.
```
bash train.sh
```

## Stage-1b (variational IB training):
```
bash vib_train.sh
```

## Stage-2 (wsi-finetuning with topK):
updating.

## Citation
If you find this repository useful, please consider giving a star :star: and citation :t-rex::
```
@InProceedings{Li_2023_CVPR,
    author    = {Li, Honglin and Zhu, Chenglu and Zhang, Yunlong and Sun, Yuxuan and Shui, Zhongyi and Kuang, Wenwei and Zheng, Sunyi and Yang, Lin},
    title     = {Task-Specific Fine-Tuning via Variational Information Bottleneck for Weakly-Supervised Pathology Whole Slide Image Classification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {7454-7463}
}
```
