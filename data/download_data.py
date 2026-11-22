"""Download the NASA Ames milling data set and check that it is the file used in the paper.

Source : Agogino A and Goebel K 2007, Milling data set, BEST Lab, UC Berkeley,
         NASA Prognostics Data Repository, NASA Ames Research Center.
         https://data.nasa.gov/dataset/milling-wear
Output : data/raw/mill.zip, data/raw/mill.mat, data/raw/Readme.pdf
Run    : python3 data/download_data.py
"""
import hashlib
import os
import sys
import urllib.request
import zipfile

URL = "https://data.nasa.gov/docs/legacy/mill.zip"
SHA256 = {
    "mill.zip": "de9c8685cb0e07b4f39459dc282b49192a3ad660e9577ff99b68dc9994e35d27",
    "mill.mat": "71486a857939d1416c86c7cf0c469d5e69c7c30495e01ec8a0e13aafd2a313cb",
}
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    os.makedirs(RAW, exist_ok=True)
    archive = os.path.join(RAW, "mill.zip")
    if not os.path.exists(archive):
        print(f"downloading {URL} (about 15 MB) ...")
        urllib.request.urlretrieve(URL, archive)
    if sha256(archive) != SHA256["mill.zip"]:
        print("WARNING: mill.zip differs from the archive used in the paper "
              "(the repository may have changed the file).")
    with zipfile.ZipFile(archive) as z:
        for name in ("mill.mat", "Readme.pdf"):
            z.extract(name, RAW)
    ok = sha256(os.path.join(RAW, "mill.mat")) == SHA256["mill.mat"]
    print("mill.mat checksum:", "OK" if ok else "DIFFERENT from the file used in the paper")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
