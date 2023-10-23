import os
import shutil


def create_dir_if_not_exist(path, file_path=True):
    """Creates a directory if does not exist.

    Args:
        path (str): The path
        file_path (bool, optional): If True, get the directory of the file provided. Defaults to True.
    """
    if file_path:
        path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)


def delete_dir_if_exists(path, file_path=True):
    """Deletes a directory if it exists.

    Args:
        path (str): The path
        file_path (bool, optional): If True, get the directory of the file provided. Defaults to True.
    """
    if file_path:
        path = os.path.dirname(path)
    if os.path.exists(path):
        shutil.rmtree(path)


def copy_dir_into(src, dst, delete_if_exists=False):  # copies and replaces
    if os.path.exists(dst):
        if delete_if_exists:
            shutil.rmtree(dst)
    # TODO: If already exists, then add/merge
    shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_file(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    create_dir_if_not_exist(dst)
    shutil.copy(src, dst)
