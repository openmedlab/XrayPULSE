"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from PIL import Image
import webdataset as wds
from xraypulse.datasets.datasets.base_dataset import BaseDataset
from xraypulse.datasets.datasets.caption_datasets import CaptionDataset
    
class OpenIDataset(CaptionDataset):

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = '{}.png'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = ann['caption']

        return {
            "image": image,
            "caption":caption,
            "image_id": self.img_ids[ann["image_id"]],
        }

