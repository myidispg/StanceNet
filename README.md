# StanceNet

This repository is my implementation of the OpenPose paper in PyTorch. Link to the paper: [Paper](https://arxiv.org/abs/1611.08050)
The original implementation in Caffe is [here](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation). <br> 
It is currently a work in progress. Most of the code is completed and I only have to make the parts interactive. It is working for videos as of now.

This repository holds the code for my implementation of OpenPose paper. This is currently a work in progress (including this README).

**None**: Download the dataset and put the annotations, train and validation images in the Coco_Dataset directory in the repository directory. The file path should be like this:\
**1: Train images-** `StanceNet/Coco_Dataset/train2017/{images}`\
**2: Validation images-** `StanceNet/Coco_Dataset/val2017/{images}`\
**3: Annotations-** `StanceNet/Coco_Dataset/annotations/person_keypoints_train2017.json` and `StanceNet/Coco_Dataset/annotations/person_keypoints_val2017.json`

## Important Note
To train the model, the dataloader depends upon `pycocotools` which can be installed using `pip install pycocotools`

## The model architecture
As per the paper, the model uses the first 10 layers of a pre-trained VGG-19 network as feature extractor. It is followed by 6 stages of Parts Affinity Fields and Heatmap generation fields. The stages were reduced to make sure the model could be trained on my hardware. 
This is one of the reason why the results were not at the level of the original paper.
![Model Architectue](https://github.com/myidispg/StanceNet/blob/master/readme_media/model_architecture.png)

## Using the system on a video
The system can be used to detect the keypoints of a person in all the frames of a video. To use this feature, execute the following command:<br>
`python detect_video.py {path_to_video_file}`<br>
The processed image will be saved to the same path with '_keypoints' appended to its name.

## Explanation of the system
The system used Confidence Maps and Parts Affinity Fields to detect the keypoints and limbs in a person. This makes it a bottom up approach and it works well for Multi-Person images too. Also, this makes the running time invariant to the number of people in tha images.<br>
![Parts and Limbs Skeleton](https://github.com/myidispg/StanceNet/blob/master/readme_media/parts_and_skeleton.png)<br>
**Confidence Maps**: A Confidence Map is a 2D representation of the belief that a particular body part can be located in any given pixel.<br>
There are J confidence maps for an image where J is the number of body joints that are to be detected. In this case, J=18. The COCO dataset has 17 keypoints, but I added a neck joint midway between the Left and Right shoulder.<br>
**Parts Affinity Fields**: A Part Affinity Field (PAF) is a set of flow fields that encodes unstructured pairwise relationships between body parts. <br>
Each pair of body parts has a (PAF), i.e neck, nose, elbow, etc,. In this system, I use 19 limbs and hence there are 19 PAFs for each image. Each PAF map contains a unit vector for that type of limb. For example, if we have a PAF for the elbow-arm limb, then for all the pixel location in the image that lie on this limb, the PAF is a 2-D unit vector from that start to end of the limb. The process to determine these pixel locations are explained very well in the paper.<br>
**Greedy Parsing of the Network Outputs**: During inference, the output Confidence Maps and PAFs are processed to get the locations of the Joints and limbs respectively. The Confidence Maps are used to determine the location of the joints. Each joint is uniquely labelled. The joint locations are nothing but the peaks of the corresponding joint heatmap.<br>
The detected joints list can be used to detect the limbs. for a given image, we have located a set of neck candidates and a set of right hip candidates. For each neck there is a possible association, or connection candidate, with each one of the right hips. So, what we have, is a complete bipartite graph, where the vertices are the part candidates, and the edges are the connection candidates. <br>
This is where PAFs enter the pipeline. I compute the line integral along the segment connecting each couple of part candidates, over the corresponding PAFs (x and y) for that pair. The line integral will give each **connection a score**, that will be saved in a weighted bipartite graph and allowed me to solve the assignment problem.<br>
I sort all the detected connections on the basis of the connection score. The connection with the highest score is a final connection. Moving on the next connection, I simply check that if no joints of the connections have been assigned to a limb before, it is a final connection. I do this for all the connections and we get the final list of all the joints.
