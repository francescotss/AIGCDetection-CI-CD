import os, logging, json, base64
import argparse
from io import BytesIO
from PIL import Image

import torch
from torchvision.transforms.functional import pil_to_tensor, normalize, crop
from utils.model_loader import load_models


def parse_args():
    parser = argparse.ArgumentParser("score")
    parser.add_argument("--input_model", default='')
    parser.add_argument("--raw_data")

    args = parser.parse_args()

    os.environ["AZUREML_MODEL_DIR"] = args.input_model
    
    print(args)
    return args



def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model_output/model.pth"
    )

    _, model = load_models(model_path, TrainMode=False)
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("request received")
    image = json.loads(raw_data)["image"]
    image = Image.open(BytesIO(base64.b64decode(image))).convert('RGB')
    image = pil_to_tensor(image).float()
    image = crop(image, top=0, left=0, height=128, width=128)
    image = normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    logging.info("Request processed")
    return predicted.item()


if __name__ == "__main__":

    args = parse_args()
    init()
    run(args.raw_data)
