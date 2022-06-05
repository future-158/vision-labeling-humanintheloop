from pathlib import Path
from itertools import chain

def rename_folder():
    # why rename folder
    # 1. for lexical sort
    # 2. not sure. but some id2label variation not working in '01' case.
    for folder in chain(Path('data/train').iterdir(), Path('data/test').iterdir()):
        newname = folder.parent / f'item_{int(folder.stem):02d}'
        print(folder.name, newname)
        folder.rename(newname)

def manual_inspection():
    """
    later implement using streamlit. not colab friendly.
    check by eye.
    """
    pass

def auto_inspection():
    """
    - exp/clip/finetune.py
    clip zeroshot prediction accuracy가 
    train / test set에 동일하게 나오므로, 
    일단 dataset shift는 없다고 봐도 무방함
    """
    pass


def class_imbalance_check():
    """
    for each split, class pair:
        check how many image it have.
    """
    pass

def image_quality_check():
    """
    check how many image are rgb
    check image size
    """
