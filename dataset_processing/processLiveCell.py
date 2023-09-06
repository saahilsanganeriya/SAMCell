import argparse
from enum import Enum
import cv2
import os
from tqdm import tqdm
import numpy as np
import pickle
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class DatasetType(Enum):
    TRAIN_2_PERCENT = 'annotations/LIVECell_dataset_size_split/0_train2percent.json'
    TRAIN_4_PERCENT = 'annotations/LIVECell_dataset_size_split/1_train4percent.json'
    TRAIN_5_PERCENT = 'annotations/LIVECell_dataset_size_split/2_train5percent.json'
    TRAIN_25_PERCENT = 'annotations/LIVECell_dataset_size_split/3_train25percent.json'
    TRAIN_50_PERCENT = 'annotations/LIVECell_dataset_size_split/4_train50percent.json'
    TRAIN_FULL = 'annotations/LIVECell/livecell_coco_train.json'
    VAL = 'annotations/LIVECell/livecell_coco_val.json'
    TEST = 'annotations/LIVECell/livecell_coco_test.json'

class DatasetPreprocessor:
    def __init__(self, base_folder:str, size:int, dataset_type:DatasetType = DatasetType.TRAIN_50_PERCENT, output_folder:str = 'output'):
        self.base_folder = base_folder
        self.size = size
        self.dataset_type = dataset_type
        self.output_folder = output_folder

        self.train_imgs = self.base_folder + '/images/livecell_train_val_images'
        self.test_imgs = self.base_folder + '/images/livecell_test_images'

        #create output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def process(self):
        print("Processing dataset...")
        self.dataset = COCO(self.base_folder + '/' + self.dataset_type.value)

        self.imgs = self.dataset.loadImgs(self.dataset.getImgIds())
        self.info = self.dataset.dataset['info']
        self.process_images()
        self.process_anns()

    def process_images(self):
        img_list = []
        for img in tqdm(self.imgs, desc=f'processing {len(self.imgs)} images'):
            #get numpy array of image
            if self.dataset_type == DatasetType.TEST:
                img_numpy = cv2.imread(self.test_imgs + '/' + img['file_name'], cv2.IMREAD_COLOR)
            else:
                img_numpy = cv2.imread(self.train_imgs + '/' + img['file_name'], cv2.IMREAD_COLOR)
            img_list.append(img_numpy)

        #convert list to numpy array
        self.img_array = np.array(img_list)

        #save to npy file in output folder
        np.save(self.output_folder + '/imgs.npy', self.img_array)

    def _fast_ann_to_mask(self, seg, ann_numpy, j):
        #convert to numpy array of points
        seg = np.array(seg, dtype=np.int32)
        seg = seg.reshape((-1, 2))

        #draw polygon
        cv2.fillPoly(ann_numpy, [seg], j, lineType=cv2.LINE_4)


    def process_anns(self):
        ann_list = []
        for i in tqdm(range(len(self.imgs)), desc=f'processing {len(self.imgs)} annotations'):
            ann_ids = self.dataset.getAnnIds(imgIds=self.imgs[i]['id'])
            anns = self.dataset.loadAnns(ann_ids)
            ann_numpy = np.zeros((2, self.img_array[i].shape[0], self.img_array[i].shape[1]), dtype=np.int16)
            j = 1
            for ann in anns:
                if ann['iscrowd'] != 0:
                    mask = self.dataset.annToMask(ann).T
                    ann_numpy[1, mask == 1] = j
                else:
                    mask = self.dataset.annToMask(ann)
                    ann_numpy[0, mask == 1] = j
                    # self._fast_ann_to_mask(ann['segmentation'], ann_numpy[0], j)

                j += 1
                
            ann_list.append(ann_numpy)

        print(len(ann_list), len(self.imgs))

        np.save(self.output_folder + '/anns.npy', ann_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--base", help="LIVECell_dataset_2021 folder path", default='../../LIVECell/LIVECell_dataset_2021')
    parser.add_argument("-s", "--size", help="size of the image", default=1024)
    parser.add_argument("-o", "--out", help="output folder", default='output')
    args = parser.parse_args()

    dataset_preprocessor = DatasetPreprocessor(args.base, args.size, output_folder=args.out)
    dataset_preprocessor.process()

if __name__ == "__main__":
    main()
