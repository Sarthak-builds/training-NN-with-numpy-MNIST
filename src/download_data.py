import requests
import os

data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "training_labels": "train-labels-idx1-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:10.0) Gecko/20100101 Firefox/10.0"
}
request_opts = {"headers": headers, "params": {"raw": "true"}}

data_dir = "../data"
os.makedirs(data_dir, exist_ok=True)
base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

def download():
    for fname in data_sources.values():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print("Downloading:", fname)
            resp = requests.get(base_url + fname, stream=True, **request_opts)
            resp.raise_for_status()
            with open(fpath, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=128):
                    fh.write(chunk)

if __name__ == "__main__":
    download()

    # .gz files because the dataset is gzip-compressed to save bandwidth.