# Simultaneous Noise Reduction and Layer Segmentation for Visible Light Optical Coherence Tomography in Human Retina
![Teaser: DenoiSeg](figs/dataset_1.png)
(preprint) Tianyi Ye, Jingyu Wang, Ji Yi. “Simultaneous Noise Reduction and Layer Segmentation for Visible Light
Optical Coherence Tomography in Human Retina”, [bioRxiv 2022.11.25.518000](https://www.biorxiv.org/content/10.1101/2022.11.25.518000v1)

## VIS-OCT dataset with noisy-clean pairs and retinal boundary delineation

This paper presents the first VIS-OCT retina image dataset for further machine-learning research. The data include retinal B-scans obtained on our 2nd Gen dual-channel [VIS-OCT system](https://www.biorxiv.org/content/10.1101/2022.10.05.511048v1) for 12 subjects, all displayed normal-appearing VIS-OCT images with varying image quality. The provided dataset includes noisy-clean image pairs, wherein the clean image is obtained by averaging 16 or 32 B-scans at the same position of the retina. It also includes 10 manually delineated retinal boundaries, each of which is individually reviewed and edited. The dataset consists of 105 B-scans with a size of 512×512 in diverse positions of the retina.

The dataset can be downloaded at [here](https://drive.google.com/drive/folders/1y3SSFjytEWdNSaGI7udKQo2Q-tcsFHBm?usp=share_link).
<br/>The pretrained models with 25% and 50% annotation used in paper are available at [here](https://drive.google.com/drive/folders/18nZEOx4yq5V9Kuk9JZTolLQriQfwULVn?usp=sharing).

## DenoiSegOCT
The code is modified from [DenoiSeg](https://arxiv.org/abs/2005.02987).  

<br/>The [Experiments notebook](https://github.com/TianyiYe98/DenoisegOCT/blob/main/Experiments.ipynb) demonstrates the preprocessing, data preparation, training and visualization of DenoiSegOCT. 
