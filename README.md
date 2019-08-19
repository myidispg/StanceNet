# StanceNet

This repository is my implementation of the OpenPose paper in PyTorch. It is currently a work in progress. Most of the code is completed and I only have to make the parts interactive. It is working for videos as of now.

This repository holds the code for my implementation of OpenPose paper. This is currently a work in progress (including this README).

**None**: Download the dataset and put the annotations, train and val images in the Coco_Dataset directory in the repository directory. The file path should be like this:\
**1: Train images-** `StanceNet/Coco_Dataset/train2017/{images}`\
**2: Validation images-** `StanceNet/Coco_Dataset/val2017/{images}`\
**3: Annotations-** `StanceNet/Coco_Dataset/annotations/person_keypoints_train2017.json` and `StanceNet/Coco_Dataset/annotations/person_keypoints_val2017.json`

## Using the system on a video
The system can be used to detect the keypoints of a person in all the frames of a video. To use this feature, execute the following command:


## File and their purposes
**constants.py**: Contains all the constant variables used throughout the project.<br>
**helper.py**: Contains all the helper functions that are used at various locations in the project.<br}
**explore_data.py**: Contains some code which I used to explore the COCO Dataset.<br>
**join_keypoints_create_pickle.py**: Converts the annotation to Python dictionaries, removes non-labelled images from the dataset, change names so that the names are sequential (000000000000.jpg, 000000000001.jpg etc.) and finally save the annotation dictionaries to pickle files.<br>
**resize_images_adjust_labels.py**: The dataset images were not of uniform shapes. Hence, resized them to 224x224 pixels and made adjustment to the labels accordingly.<br>
