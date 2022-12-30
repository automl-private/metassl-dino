from pathlib import Path
import random

def create_imagenet_subset(percentage):
    # path preparation
    name = str(percentage) + "percent"
    path_to_get_from_train = Path("/data/datasets/ImageNet/imagenet-pytorch/train")
    path_to_save_to_train = "/work/dlclarge2/wagnerd-metassl-experiments/datasets/ImageNetSubset/" + name + "/train"
    path_to_get_from_val = Path("/data/datasets/ImageNet/imagenet-pytorch/val")
    path_to_save_to_val = "/work/dlclarge2/wagnerd-metassl-experiments/datasets/ImageNetSubset/" + name + "/val"
    Path(path_to_save_to_train).mkdir(parents=True, exist_ok=True)
    Path(path_to_save_to_val).mkdir(parents=True, exist_ok=True)
    
    # get list from all labels (folders) from path_to_get_from_train
    labels = list(path_to_get_from_train.iterdir()) 
    label_list = random.sample(labels, int(1000/percentage))
    
    for element in label_list:
        Path(path_to_save_to_train, element.name).symlink_to(element)
        Path(path_to_save_to_val, element.name).symlink_to(path_to_get_from_val / element.name)

if __name__ == '__main__':
    percentage = 10
    create_imagenet_subset(percentage)

