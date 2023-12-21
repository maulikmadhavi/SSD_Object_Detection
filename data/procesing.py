# This file is used to read LMDB  data and convert into images
import os


# Read LMDB data

import pickle

import cv2
import lmdb
import numpy as np
import torch
import torchvision.ops.boxes as bx
from torch.utils.data import Dataset
from tqdm import tqdm


def unpack_img2(buf, iscolor=1):
    img = np.frombuffer(buf, dtype=np.uint8)
    assert cv2 is not None
    img = cv2.imdecode(img, iscolor)
    return img


# ============ Load Images and Labels ============
class LoaderBaseClass(Dataset):
    def __init__(self, path):
        self.get_env_len(path)

    def __len__(self):
        return self.length

    @staticmethod
    def collate_fn(batch):
        (img, label, path, shapes) = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i
        return (torch.stack(img, 0), torch.cat(label, 0), path, shapes)

    def get_env_len(self, path):
        self.env = lmdb.open(
            path,
            subdir=os.path.isdir(path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
            self.keys = pickle.loads(txn.get(b"__keys__"))

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        self.unpacked = pickle.loads(byteflow)
        return self.unpacked
        # There are six return values:
        # 1. img: the image, a tensor of shape (3, H, W)
        # 2. target: list of dicts, each has keys: bbox, category_id, image_id
        # 3. path: the path of the image file
        # 4 & 5. shapes: (height, width) of the image
        # 6: image_id of the image, a tensor of shape (1,)

def debug_function():
    train_data = "/home/maulik/Documents/detection-sdk/data/train.lmdb"
    data = LoaderBaseClass(train_data)
    print(data)
    print(data[0])

    index = 9
    img, tar, _, _, _, _ = data[index]
    print(bx.box_convert(torch.tensor(tar[index]["bbox"]), "xywh", "xyxy"))

    img = unpack_img2(img)

    img = produce_debug_image(img, tar)
    cv2.imwrite("test.jpg", img)


def produce_debug_image(img, tar):
    h, w = img.shape[0], img.shape[1]
    whwh = torch.tensor([w, h, w, h])
    label_to_color = {"0": (255, 0, 0), "1": (0, 255, 0), "2": (0, 0, 255)}
    for each_box in tar:
        box = torch.tensor(each_box["bbox"])
        box = (bx.box_convert(box, "xywh", "xyxy") * whwh).int().tolist()
        x1, y1, x2, y2 = box
        label = each_box["category_id"][0]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), label_to_color[label], 2)
    return img


def _lmdb2_folder_convert(
    lmdb_path: str, folder_path: str, debug: bool = False, debug_index: int = 400
):
    img_path = os.path.join(folder_path, "images")
    label_path = os.path.join(folder_path, "labels")
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    if debug:
        debug_path = os.path.join(folder_path, "debug")
        os.makedirs(debug_path, exist_ok=True)

    data = LoaderBaseClass(lmdb_path)
    for i, content in tqdm(enumerate(data), total=len(data)):
        img, tar, _, _, _, img_id = content
        img = unpack_img2(img)
        h, w = img.shape[0], img.shape[1]
        whwh = torch.tensor([w, h, w, h])
        cv2.imwrite(os.path.join(img_path, f"{img_id}.jpg"), img)
        with open(os.path.join(label_path, f"{img_id}.txt"), "w") as f:
            for each_box in tar:
                box = torch.tensor(each_box["bbox"])
                box = (bx.box_convert(box, "xywh", "xyxy") * whwh).tolist()
                x1, y1, x2, y2 = box
                label = each_box["category_id"][0]
                f.write(f"{label} {x1} {y1} {x2} {y2}\n")

        if debug and (i + 1) % debug_index == 0:
            img = produce_debug_image(img, tar)
            cv2.imwrite(os.path.join(debug_path, f"{img_id}.jpg"), img)

if __name__ == "__main__":
    for mode in ["train", "val"]:
        print(f"Converting {mode} data")
        LMDB_PATH = f"/home/maulik/Documents/detection-sdk/data/{mode}.lmdb"
        FOLDER_PATH = f"/home/maulik/Desktop/SSD_Object_Detection/DATA/{mode}"
        _lmdb2_folder_convert(LMDB_PATH, FOLDER_PATH, debug=True, debug_index=400)