
import argparse
import base64
import json
import requests

def parse_args():
    parser = argparse.ArgumentParser("test_client")
    parser.add_argument("--endpoint_url")
    parser.add_argument("--endpoint_key")
    parser.add_argument("--img")

    args = parser.parse_args()
    return args

def test_client(endpoint_url, endpoint_key, img):
    data = {}
    with open(img, mode='rb') as file:
        img = file.read()
    data['image'] = base64.encodebytes(img).decode('utf-8')
    raw_data = json.dumps(data)

    headers = {'Content-Type':'application/json',
               "Authorization": f"Bearer {endpoint_key}"
               }
    res = requests.post(endpoint_url, data=raw_data, headers=headers)

    print(res.status_code, res.reason)
    if res.json() == 0:
        print("The image is real")
    else:
        print("The image is AI generated")





if __name__ == "__main__":

    args = parse_args()
    test_client(args.endpoint_url,
                args.endpoint_key,
                args.img)