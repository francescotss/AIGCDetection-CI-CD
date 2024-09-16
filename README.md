# Continuous Fake Media Detection: Adapting Deepfake Detectors to New Generative Techniques

<div align="left">

  [![DOI](https://img.shields.io/badge/DOI-10.1016/j.cviu.2024.104143-blue.svg)](https://doi.org/10.1016/j.cviu.2024.104143)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

 

</div>

This repository contains the code from the paper "Continuous Fake Media Detection: Adapting Deepfake Detectors to New Generative Techniques". It explores the application of continual learning methods like Knowledge Distillation and Elastic Weight Consolidation to improve deepfake detection over evolving media types. The project includes a proposed CI/CD pipeline for integrating continuous learning to maintain detector performance as new generative techniques emerge.

 ![mlops](https://github.com/user-attachments/assets/b5feedda-914a-4d49-9b62-cf27949ff6a7)
  

### Abstract

> Generative techniques continue to evolve at an impressively high rate, driven by the hype about these technologies. This rapid advancement severely limits the application of deepfake detectors, which, despite numerous efforts by the scientific community, struggle to achieve sufficiently robust performance against the ever-changing content. To address these limitations, in this paper, we propose an analysis of two continuous learning techniques on a Short and a Long sequence of fake media. Both sequences include a complex and heterogeneous range of deepfakes (generated images and videos) from GANs, computer graphics techniques, and unknown sources. Our experiments show that continual learning could be important in mitigating the need for generalizability. In fact, we show that, although with some limitations, continual learning methods help to maintain good performance across the entire training sequence. For these techniques to work in a sufficiently robust way, however, it is necessary that the tasks in the sequence share similarities. In fact, according to our experiments, the order and similarity of the tasks can affect the performance of the models over time. To address this problem, we show that it is possible to group tasks based on their similarity. This small measure allows for a significant improvement even in longer sequences. This result suggests that continual techniques can be combined with the most promising detection methods, allowing them to catch up with the latest generative techniques. In addition to this, we propose an overview of how this learning approach can be integrated into a deepfake detection pipeline for continuous integration and continuous deployment (CI/CD). This allows you to keep track of different funds, such as social networks, new generative tools, or third-party datasets, and through the integration of continuous learning, allows constant maintenance of the detectors.

## Setup


### Dependencies

```
pip install -r requirements.txt
```

### Dataset

1. Prepare the dataset containing real and generated images using the following structure:

```
<dataset1>/train/
|--0_real/
   |--img1.png
   |--img2.png
|--1_fake/
   |--img1.png
   |--img2.png

<dataset1>/val/
|--0_real/
   |--img1.png
   |--img2.png
|--1_fake/
   |--img1.png
   |--img2.png

<dataset1>/test/
|--0_real/
   |--img1.png
   |--img2.png
|--1_fake/
   |--img1.png
   |--img2.png

<dataset2>/train
|-- ...
<dataset2>/test
|-- ...
<dataset2>/val
|-- ...
...
```


### Train
#### Quickstart

Knowledge Distillation from Task A (Dataset 1) to Task A+B (Dataset 1 & Dataset 2)

```
python train.py --network ResNet --input_model models/KD_resnet/taskA --output_dir models/KD_resnet --source_datasets datasets/dataset1 --target_dataset datasets/dataset2
```

**Tip:** Check [notebooks/train_local.ipynb](notebooks/train_local.ipynb) for examples and training scripts


## Citation

```
@article{tassone2024continuous,
  title={Continuous Fake Media Detection: Adapting Deepfake Detectors to New Generative Techniques},
  author={Tassone, Francesco and Maiano, Luca and Amerini, Irene},
  journal={Computer Vision and Image Understanding},
  year={2024},
  publisher={Elsevier}
}
```
