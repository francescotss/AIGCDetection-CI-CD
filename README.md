# Continuous fake media detection: adapting deepfake detectors to new generative techniques

![mlops](https://github.com/user-attachments/assets/b5feedda-914a-4d49-9b62-cf27949ff6a7)


This project aims to incorporate continuous learning techniques into a real-world, end-to-end deepfake detection system that supports continuous integration and continuous delivery/deployment (CI/CD). Our implementation is a simple yet effective MLOps pipeline design that enables the end-to-end development of continuously trained and monitored intelligent detectors using a limited dataset.

## Architecture

*Stay tuned*

## Setup

#### 1. Dependancies

*Stay tuned*

#### 2. Prepare data

1. Prepare the dataset containing real and generated images from the new generative source (e.g. Stable Diffusion, DALL-E ...) using the following structure:

```
dsname/train/
---0_real/
------img1.png
------img2.png
---1_fake/
------img1.png
------img2.png

dsname/val/
---0_real/
------img1.png
------img2.png
---1_fake/
------img1.png
------img2.png

dsname/test/
---0_real/
------img1.png
------img2.png
---1_fake/
------img1.png
------img2.png
```



#### 3. Train

*Stay tuned*

#### 4. Validate and deploy

*Stay tuned*
