**an explanation of each file in the repository is below:**

apt_packages.txt: file created after sudo apt-related equivalent of pip freeze > requirements.txt

coco_labels.txt: annotations for primary data in dataset (must be in assets folder along with colours.txt and imagenet_labels.txt

colours.txt: another asset

imagenet_labels.txt: another asset

opod.py: main python script that facilitates camera functioning: modified to print detected objects to terminal only if above a certain confidence threshold. furthermore, 
when an animal is detected, it prints out a safety tip pertaining to that animal to terminal. also, camera feed interface has been slightly modified to remove borders and other 
info

requirements.txt: result of pip freeze
