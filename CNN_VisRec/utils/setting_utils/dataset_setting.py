
from utils.dataset_utils import galaxy


def training_dataset_defaults():
    """
    Defaults for training galaxy dataset.
    """
    return dict(
        dataset="galaxy",
        data_dir="",
        data_info_list_path="",
        batch_size=1,
        random_flip=False,
        is_distributed=True,
        num_workers=0,
    )


def create_training_dataset(
        dataset,
        data_dir,
        data_info_list_path,
        batch_size,
        random_flip,
        is_distributed,
        num_workers,
):


    if dataset == "galaxy":
        load_data = galaxy.load_data
    else:
        raise ValueError
    return load_data(
        data_dir=data_dir,
        data_info_list_path=data_info_list_path,
        batch_size=batch_size,
        random_flip=random_flip,
        is_distributed=is_distributed,
        is_train=True,
        post_process=None,
        num_workers=num_workers,
    )


def test_dataset_defaults():

    return dict(
        dataset="galaxy",
        data_dir="",
        data_info_list_path="",
        batch_size=1,
        random_flip=False,
        is_distributed=True,
        num_workers=0,
    )


def create_test_dataset(
        dataset,
        data_dir,
        data_info_list_path,
        batch_size,
        random_flip,
        is_distributed,
        num_workers,
):

    if dataset == "galaxy":
        load_data = galaxy.load_data
    else:
        raise ValueError
    return load_data(
        data_dir=data_dir,
        data_info_list_path=data_info_list_path,
        batch_size=batch_size,
        random_flip=random_flip,
        is_distributed=is_distributed,
        is_train=False,
        post_process=None,
        num_workers=num_workers,
    )
