from .base_data_set import BaseDataset
from typing import Tuple, TypedDict, Optional
import os
from src.utilities.os_utilities import copy_dir, copy_file


class BACHSettings(TypedDict):
    download_link: Optional[os.PathLike]


class BACH(BaseDataset):
    def __init__(self, BACH_settings: BACHSettings) -> None:
        super().__init__()
        self.BACH_settings = BACH_settings
        self.class_names = ["Benign", "InSitu", "Invasive", "Normal"]

    def get_data_set_name(self) -> str:
        return "BACH"

    def get_data_set_zip_name(self) -> str:
        return "BACH.zip"

    def fetch_zipped(self, zip_tree: str):
        if (self.BACH_settings['download_link'] is None):
            raise Exception("No download link provided for BACH")
        else:
            raise NotImplementedError("Download link not implemented yet")

    def process(self, unzipped_tree: str, processed_tree: str):
        # Transfer files
        photos_folder = os.path.join(
            unzipped_tree, self.get_data_set_name(), "ICIAR2018_BACH_Challenge", "Photos")
        for class_name in self.class_names:
            copy_dir(os.path.join(photos_folder, class_name),
                     os.path.join(processed_tree, self.get_data_set_name()))
        # TODO:
        pass

    def train_test_split(self, train_location: str, test_location: str, train_test_split: Tuple[float, float]):
        # TODO
        pass
