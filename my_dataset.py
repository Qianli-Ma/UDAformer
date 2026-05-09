import os

import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MyDataSet(Dataset):
    def __init__(self, file_path, raw_folder="raw-890", reference_folder="reference-890"):
        if not os.path.isdir(file_path):
            os.makedirs(file_path, exist_ok=True)

        self.file_path = file_path
        self.raw_dir = os.path.join(file_path, raw_folder)
        self.reference_dir = os.path.join(file_path, reference_folder)

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.reference_dir, exist_ok=True)

        self.raw_list = self._list_images(self.raw_dir)
        self.reference_list = self._list_images(self.reference_dir)

        if not self.raw_list:
            raise ValueError(f"No images found in raw folder: {self.raw_dir}")
        if not self.reference_list:
            raise ValueError(f"No images found in reference folder: {self.reference_dir}")
        if len(self.raw_list) != len(self.reference_list):
            raise ValueError(
                "Raw and reference folder sizes do not match: "
                f"{len(self.raw_list)} != {len(self.reference_list)}"
            )

        self.transforms = transforms.Compose([transforms.ToTensor()])

    def _list_images(self, directory):
        return sorted(
            name
            for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))
        )

    def __getitem__(self, index):
        raw_path = os.path.join(self.raw_dir, self.raw_list[index])
        reference_path = os.path.join(self.reference_dir, self.reference_list[index])
        raw_image = self._read_convert_image(raw_path)
        reference_image = self._read_convert_image(reference_path)
        return raw_image, reference_image

    def _read_convert_image(self, image_name):
        image = cv2.imread(image_name)
        if image is None:
            raise ValueError(f"Failed to read image: {image_name}")
        image = cv2.resize(image, (256, 256))
        image = self.transforms(image).float()
        return image

    def __len__(self):
        return len(self.raw_list)
