#! python3
# -*- encoding: utf-8 -*-

import os
import numpy as np
import logging
import time

from collections import OrderedDict
from typing import List, Optional

import cv2
import torch
import torch.utils.data as data
import pandas as pd


IMG_EXTENSIONS = ('.jpg*', '.jpeg*', '.png', '.ppm', '.bmp', '.pgm',
 '.tif', '.tiff', '.webp')
IMG_EXTENSIONS.extend([x.upper() for x in IMG_EXTENSIONS])


def cv2_loader(path):
    try:
        image = cv2.imread(path)[:, :, ::-1] #RGB image
        return image
    except: 
        logging.info(f"Cannot read image: {path}")
        return None


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        raise(RuntimeError('No Module named accimage'))
    else:
        return cv2_loader(path)


def collate_fn(batch):
    batch = filter(lambda elem: elem is not None, batch)
    return torch.utils.data.dataloader.default_collate( list(batch) )



class MultiLabelDataset(data.Dataset):
    """Pytorch multi label dataset for label saved in csv."""
    def __init__(
        self,
        csv_file,
        class_names: List[str],
        is_train: bool,
        train_split: Optional[float] = 0.,
        transform: Optional[bool] = None,
        random_seed: Optional[int] = None, #42
        loader=default_loader,
        sample: Optional[str] = None,
        sample_class_names: Optional[List[str]] = None
    ) -> None:
        """
        Args:
            csv_file ([type]): Path to the csv file with annotations.
            class_names (List[str]): Class names. Should be included in
                 the csv_file column names
            is_train (bool): Use together with train_split to control
                 the training indices.
            train_split (Optional[float], optional): Use together with
                 train_split to control the training indices. Defaults to 0..
            transform (Optional[bool], optional): Transform to be applied.
                 Defaults to None.
            random_seed (Optional[int], optional): Random seed. Defaults
                 to None.
            sample (Optional[str], optional('oversample')): Sample strategy
                 for imbalanced dataset. Support 'oversample' only for now.
                  Defaults to None.
            sample_class_names (Optional[List[str]], optional): Use together
                 with 'sample' argument. Defaults to None, which means
                  oversample the half less samples.

        Raises:
            NotImplementedError: [description]
        """
        logging.info("=> Initialize dataset...")
        t1 = time.time()
        self.roofs_frame = pd.read_csv(csv_file, low_memory=False)#, nrows=1280)
        self.roofs_frame = self.roofs_frame[['image_path']+class_names].dropna(subset=class_names)
        self.class_names = class_names
        self.transform = transform
        self.loader = loader
        self.sample = sample
        self.sample_class_names = sample_class_names
        self.num_classes = len(self.class_names)
        self.classes = dict(zip(range(self.num_classes), class_names))
        self.img_indice = np.arange(0, self.roofs_frame.shape[0])

        if random_seed:
            np.random.seed(random_seed)
        np.random.shuffle(self.img_indice)

        # Create train/validation twice using same random_seed
        # only if you need to seperate dataset yourself
        self.last_train_sample = int(len(self.img_indice) * train_split)
        if is_train:
            self.img_indice = self.img_indice[:self.last_train_sample]
        else:
            self.img_indice = self.img_indice[self.last_train_sample:]
        logging.info(f"*** first 10 indices are {self.img_indice[:10]}")

        # Calc label counts
        self.class_sample_counts = {}
        for class_name in self.class_names:
            self.class_sample_counts[class_name] = len(
                self.roofs_frame.iloc[self.img_indice][self.roofs_frame[class_name] > .5]
            )
        logging.info( str(self.class_sample_counts) )

        # TODO sample for addressing label imbalance problem
        #     only realize 'oversample' for now
        if self.sample == 'oversample':
            sorted_cls_sample_cnts = OrderedDict(
                sorted(self.class_sample_counts.items(), 
                key=lambda item: item[1])
            )
            maxcount = list(sorted_cls_sample_cnts.items())[-1][1]
            if self.sample_class_names is None:
                # Only oversample the 1/2 less labels if not given 'sample_class_names'
                self.sample_class_names = list(
                    sorted_cls_sample_cnts.keys())[:int(0.5*len(self.class_names))]
            for class_name in self.sample_class_names:
                class_cnt = sorted_cls_sample_cnts[class_name]
                gapnum = maxcount - class_cnt
                #print(gapnum)
                temp_df = self.roofs_frame.iloc[
                    np.random.choice(
                        np.where(self.roofs_frame[class_name] > .5)[0],
                        size=gapnum
                        )]
                self.roofs_frame = self.roofs_frame.append(temp_df,ignore_index=True)

            self.roofs_frame = self.roofs_frame.sample(frac=1).reset_index(drop=True)
            self.img_indice = np.arange(0, self.roofs_frame.shape[0])
            # Re-Calc label counts
            self.class_sample_counts = {}
            for class_name in self.class_names:
                self.class_sample_counts[class_name] = len(
                    self.roofs_frame.iloc[self.img_indice][self.roofs_frame[class_name]>.5]
                )
            logging.info(str(
                self.class_sample_counts
            ))
        elif self.sample is not None:
            raise NotImplementedError(
                "Only realize 'oversample' option. \
                Please use 'oversample' as 'sample' argument \
                if you want to try.")

        logging.info(
            "<= Dataset prepared well in {:.2f} s...".format(time.time()-t1)
        )

    def __len__(self):
        return len(self.img_indice)

    def __getitem__(self, idx):
        try:
            img_idx = self.img_indice[idx]
            row = self.roofs_frame.iloc[img_idx]
            img_info = dict(
                image_path=row['image_path'],
                image_name=row['image_path'].split('/')[-1],
                soft_label=row[self.class_names].to_numpy(dtype=np.float32)
            )
            img = self.loader(img_info['image_path'])

            if self.transform:
                sample = self.transform(image=img)['image'] #albumentations apply
            else:
                sample = img
            return dict(
                image=sample, 
                imgpath=img_info['image_path'],
                soft_label=img_info['soft_label'],
            )
        except Exception as e:
            logging.debug(f"Expection @Index#{idx}: {e}")
            return None
