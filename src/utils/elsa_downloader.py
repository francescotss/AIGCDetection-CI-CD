from datasets import load_dataset
from PIL import Image
import os, requests
from io import BytesIO
import argparse

def parse_args():
    parser = argparse.ArgumentParser("downloader")
    parser.add_argument("-d", "--dest_path", type=str)
    parser.add_argument("--source_split", help='train/validation')
    parser.add_argument("--dest_split", help="train/val/test")
    parser.add_argument("--to_save", type=int)
    parser.add_argument("--skip", default=0, type=int)

    args = parser.parse_args()
    return args


def download(args):
    des_path = args.dest_path
    source_split = args.source_split
    dest_split = args.dest_split
    to_save = args.to_save
    skip = args.skip

    os.makedirs(des_path+"/test/0_real", exist_ok=True)
    os.makedirs(des_path+"/test/1_fake", exist_ok=True)
    os.makedirs(des_path+"/train/0_real", exist_ok=True)
    os.makedirs(des_path+"/train/1_fake", exist_ok=True)
    os.makedirs(des_path+"/val/0_real", exist_ok=True)
    os.makedirs(des_path+"/val/1_fake", exist_ok=True)


    dataset = load_dataset("elsaEU/ELSA_D3", split=source_split, streaming=True)
    dataset = dataset.skip(skip)

    opened, saved = 0, 0
    save_path = f"{des_path}/{dest_split}"
    for entry in dataset:
        opened += 1
        print(opened, saved)
        name = entry["id"]
        real_url = entry["url"]
        fake_img = entry["image_gen0"]

        real_extention = real_url.split("?")[0].split(".")[-1]
        if not real_extention in ("png", "PNG", "jpg","jpeg","JPEG", "JPG"):
            continue

                
        try:
            response = requests.get(real_url, timeout=2, allow_redirects=False)
            if not response.ok:
                continue
            real_img = Image.open(BytesIO(response.content))
        except:
            print("Error reading image, skipping")
            print(entry, response.content)
            continue

        if real_img.size < (244,244):
            continue

        if real_img.mode != "RGB":
            continue

        real_img.save(f'{save_path}/0_real/{name}.{real_extention}')
        fake_img.save(f'{save_path}/1_fake/{name}.{real_extention}')
        saved += 1
        print("saved", real_img.size, name, real_url)
        if saved >= to_save:
            break

if __name__ == "__main__":
    args = parse_args()
    download(args)