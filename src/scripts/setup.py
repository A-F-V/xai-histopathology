
import torch
from src.scripts.data_scripts.preprocess_data import create_instance_segmentation_mask, move_and_rename, create_semantic_segmentation_mask
from src.transforms.cell_segmentation.hover_maps import hover_map
from src.utilities.img_utilities import *
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import os

from src.datasets.PanNuke import PanNuke
from src.utilities.os_utilities import copy_dir_into, copy_file

from src.datasets.base_data_set import DatasetLoader, DatasetLoadingRules
from src.datasets.BACH import BACH


#
# def normalize(image_path):
#    img = Image.open(image_path)
#    img = normalize_he_image(ToTensor()(img), alpha=1, beta=0.15, check=singular_norm_check)
#    img = ToPILImage()(img)
#    img.save(image_path)
def setup():
    rules: DatasetLoadingRules = {"zip_tree": os.path.join("data", "raw", "zipped"),
                                  "unzip_tree": os.path.join("data", "raw", "unzipped"),
                                  "processed_tree": os.path.join("data", "processed"),
                                  "train_tree": os.path.join("data", "processed", "train"),
                                  "test_tree": os.path.join("data", "processed", "test"),
                                  "replace_zipped": False,
                                  "replace_unzipped": False,
                                  "new_split": False,
                                  "verbose": True,
                                  "train_test_split": (0.8, 0.2)}

    loader = DatasetLoader(rules)

    loader.load(BACH({"download_link": None}))

    return
    # STAIN NORMALIZE and Luminescence normalize
    if normalize:
        for folder in ["Benign", "InSitu", "Invasive", "Normal"]:
            folder_path = os.path.join(BACH_folder_final, folder)
            for image_name in tqdm(os.listdir(folder_path), desc=f"Normalizing {folder} Images - BACH"):
                img_path = os.path.join(folder_path, image_name)
                if ".tif" not in img_path:
                    os.remove(img_path)
                else:
                    normalizer.normalize(img_path, img_path)

    ###############################################################################
    # Process MoNuSeg                                                            #
    ###############################################################################

    MoNuSeg_unzipped = os.path.join(raw_tree, "unzipped",
                                    "MoNuSeg", "MoNuSeg 2018 Training Data")
    move_and_rename(MoNuSeg_unzipped,
                    {"Annotations": "annotations", "Tissue Images": "images"},
                    os.path.join(processed_tree, "MoNuSeg"))

    for image_name in tqdm(os.listdir(os.path.join(processed_tree, "MoNuSeg", "images")), desc="Extracting Annotated Masks - MoNuSeg"):
        img_path = os.path.join(
            processed_tree, "MoNuSeg", "images", image_name)
        anno_path = os.path.join(
            processed_tree, "MoNuSeg", "annotations", image_name.split(".")[0] + ".xml")
        dst_folder_sm = os.path.join(
            processed_tree, "MoNuSeg", "semantic_masks")
        dst_folder_im = os.path.join(
            processed_tree, "MoNuSeg", "instance_masks")
        create_semantic_segmentation_mask(anno_path, img_path, dst_folder_sm)
        create_instance_segmentation_mask(anno_path, img_path, dst_folder_im)

    if normalize:
        for image_name in tqdm(os.listdir(os.path.join(processed_tree, "MoNuSeg", "images")), desc="Normalizing Images - MoNuSeg"):
            img_path = os.path.join(
                processed_tree, "MoNuSeg", "images", image_name)
            normalizer.normalize(img_path, img_path)

        ###############################################################################
    # Process PanNuke                                                             #
    ###############################################################################

    PanNuke.prepare(os.path.join(unzipped_folder, 'PanNuke_orig'),
                    os.path.join(processed_tree, "PanNuke"))

    # NORM IMAGES - PanNuke

    images = np.load(os.path.join(processed_tree, "PanNuke", "images.npy"))

    def safe_norm(img):
        try:
            return tensor_to_numpy(normalize_he_image(numpy_to_tensor(img), alpha=1, beta=0.15, check=singular_norm_check))
        except:
            return img

    norm_images = [safe_norm(img)
                   for img in tqdm(images, desc="Normalizing Images - PanNuke")]
    norm_images = np.stack(norm_images, axis=0)
    if norm_images.max() <= 1:
        norm_images *= 255
    norm_images = norm_images.astype(np.uint8)
    np.save(os.path.join(processed_tree, "PanNuke", "images.npy"),
            norm_images if normalize else images)

    # GENERATE HOVER MAPS
    masks = np.load(os.path.join(processed_tree, "PanNuke", "masks.npy"))
    # last dim is mask channels, last channel is instance mask
    hv_maps = [hover_map(mask[:, :, -1].astype("int16"))
               for mask in tqdm(masks, desc="Generating HoVer Maps - PanNuke")]
    hv_maps = torch.stack(hv_maps).numpy()
    np.save(os.path.join(processed_tree, "PanNuke", "hover_maps.npy"), hv_maps)
