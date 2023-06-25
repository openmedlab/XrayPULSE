import os
from PIL import Image
import webdataset as wds
from xraypulse.datasets.datasets.base_dataset import BaseDataset
from xraypulse.datasets.datasets.caption_datasets import CaptionDataset

class MIMICDataset(CaptionDataset):

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = '{}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = ann['caption']

        return {
            "image": image,
            "caption":caption,
            "image_id": self.img_ids[ann["image_id"]],
        }