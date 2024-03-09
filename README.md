# MLOps for Deepfake Detection

This project aims to incorporate continuous learning techniques into a real-world, end-to-end deepfake detection system that supports continuous integration and continuous delivery/deployment (CI/CD). Our implementation is a simple yet effective MLOps pipeline design that enables the end-to-end development of continuously trained and monitored intelligent detectors using a limited dataset.

## Architecture

*Stay tuned*

## Setup

### Prerequisites

- Azure subscription
- Azure [service principal](https://learn.microsoft.com/en-us/powershell/azure/create-azure-service-principal-azureps?view=azps-11.3.0)

### Repo Structure

1. `.cloud` Azure Machine Learning IaC files for provisioning resources to train and deploy models
2. `.github/workflows` GitHub Actions workflows. Used to automatically trigger the training pipeline on pull requests and the deployment pipeline on merges
3. `pipeline` Azure Machine Learning pipeline definitions
4. `notebooks` Notebooks for managing dataset upload and performing tests
5. `src` Python code for train, evaluate, and deploy models

#### 1. Deploy Infrastructure

1. Add service principal credential in the repository secrets
2. Configure environment parameters in the `.cloud/config-infra-prod.yml` file
3. Run workflow

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

2. Upload the folder (data asset) on the Azure machine learning workspace using the script provided in `notebooks/manage_azure.ipynb` or AZCopy

#### 3. Train

1. Using a feature branch, configure training in `model_config.conf`
2. Push changes and create a pull request, it will trigger a workflow to run the training pipeline

#### 4. Validate and deploy

*Work in progress*
