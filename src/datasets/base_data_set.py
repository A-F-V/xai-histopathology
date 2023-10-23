from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import TypedDict
from typing import Literal, Tuple
import os
import zipfile
import tarfile
import os
from src.utilities.os_utilities import delete_dir_if_exists, copy_dir_into

# Literal type
ZipFormat = Literal["ZIP", "TAR", "NONE"]


class BaseDataset(Dataset, ABC):
    """Handles preperation and loading of data sets.
    """

    @abstractmethod
    def get_data_set_name(self) -> str:
        pass

    @abstractmethod
    def get_data_set_zip_name(self) -> str:
        pass

    # Gets the zip file from somewhere and saves it in the zip tree

    @abstractmethod
    def fetch_zipped(self, zipped_tree: str):
        pass

    def fetch_unzipped(self, zip_file: str, unzipped_tree: str):
        # Check if ends in .zip or .tar.gz
        if (zip_file.endswith(".zip")):
            zip_type = "ZIP"
        elif (zip_file.endswith(".tar.gz")):
            zip_type = "TAR"
        else:
            zip_type = "NONE"

        # If it is a zip file, extract it via zipfile
        if (zip_type == "ZIP"):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(unzipped_tree)
        # If it is a tar file, extract it via tarfile
        elif (zip_type == "TAR"):
            with tarfile.open(zip_file, 'r:gz') as tar_ref:
                tar_ref.extractall(unzipped_tree)
        else:
            raise Exception("File is not a zip or tar file.")

    # Takes the unzipped files and processes them into a list of resources
    @abstractmethod
    def process(self, unzipped_tree: str, processed_tree: str):
        pass

    # Typically done in a way that preserves certain dataset proportions
    @abstractmethod
    def train_test_split(self, train_tree: str, test_tree: str, train_test_split: Tuple[float, float]):
        pass
    # TODO:
    # Get an item


class DatasetLoadingRules(TypedDict):
    zip_tree: str
    unzip_tree: str
    processed_tree: str
    train_tree: str
    test_tree: str

    replace_zipped: bool
    replace_unzipped: bool
    new_split: bool

    verbose: bool

    train_test_split: Tuple[float, float]


class DatasetLoader:
    def __init__(self, rules: DatasetLoadingRules) -> None:
        self.rules = rules

    def load(self, dataset: BaseDataset):
        name = dataset.get_data_set_name()
        # 1) Fetch zipped
        # Check if the zipped already exists
        # If it does, check if we want to replace it
        # If we do, replace it
        zip_tree = os.path.join(self.rules['zip_tree'])
        unzipped_tree: str = os.path.join(self.rules['unzip_tree'],
                                          dataset.get_data_set_name())
        processed_tree = os.path.join(self.rules['processed_tree'],
                                      dataset.get_data_set_name())
        train_tree = os.path.join(self.rules['train_tree'],
                                  dataset.get_data_set_name())
        test_tree = os.path.join(self.rules['test_tree'],
                                 dataset.get_data_set_name())

        zip_file = os.path.join(zip_tree, dataset.get_data_set_zip_name())
        already_downloaded = os.path.exists(zip_tree)
        replace_zipped = self.rules['replace_zipped']
        self.verbose_print("Attempting to fetch: " + name)
        if (replace_zipped or not already_downloaded):

            if (already_downloaded):
                self.verbose_print("Removing old zip file: " +
                                   dataset.get_data_set_zip_name())
                delete_dir_if_exists(zip_tree)
            self.verbose_print("Downloading: " + name)
            dataset.fetch_zipped(zip_tree)
        else:
            self.verbose_print("Already downloaded: " + name)
        # 2) Fetch unzipped
        already_unzipped = os.path.exists(unzipped_tree)
        replace_unzipped = self.rules['replace_unzipped']
        self.verbose_print("Attempting to unzip: " + name)
        if (replace_unzipped or not already_unzipped):
            if (already_unzipped):
                self.verbose_print(
                    "Removing old unzipped folder: " + dataset.get_data_set_name())
                delete_dir_if_exists(unzipped_tree)
            self.verbose_print("Unzipping: " + name)
            dataset.fetch_unzipped(zip_file, unzipped_tree)
        else:
            self.verbose_print("Already unzipped: " + name)
        # 3) Process
        self.verbose_print("Process: " + name)
        dataset.process(unzipped_tree, processed_tree)
        self.verbose_print("Finished processing: " + name)

        # 4) Split into train and test
        self.verbose_print("Splitting: " + name)
        already_split = os.path.exists(
            train_tree) and os.path.exists(test_tree)
        new_split = self.rules['new_split']
        if (new_split or not already_split):
            if (already_split):
                self.verbose_print("Removing old split: " +
                                   dataset.get_data_set_name())
                delete_dir_if_exists(train_tree)
                delete_dir_if_exists(test_tree)
            self.verbose_print("Splitting: " + name)
            dataset.train_test_split(
                train_tree, test_tree, self.rules["train_test_split"])
        else:
            self.verbose_print("Already split: " + name)

    def verbose_print(self, print_str):
        if (self.rules['verbose']):
            print(print_str)
