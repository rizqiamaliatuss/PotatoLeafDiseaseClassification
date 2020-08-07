# Potato Leaf Disease Classification
Develop a system that can classify and detect leaf diseases in potato plants based on deep learning. This system can help farmers and agricultural researchers to get accurate and fast diagnose results of disease in plants, especially in potato plant.

clone : https://github.com/rizqiamaliatuss/PotatoLeafDiseaseClassification.git

This experiment consist of 3 major step :

# 1. Prepare Dataset

--- Datasets was collected from :

1. PlantVillage datasets = https://www.kaggle.com/emmarex/plantdisease
2. Potato Plantation, Malang, Indonesia.
3. Google Images = i recommend you to follow this tutorial to collect dataset from google images https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/

--- Clustering dataset 

Divide dataset into 5 class : Altenaria Solani, Healthy Leaf, Virus, Insect and Phytophthora Infestan.

\--Dataset 
   +--- Alternaria Solani
   +--- Healthy Leaf 
   +--- Virus
   +--- Insect 
   +--- Phytophthora Infestan
 
 --- Cropping image 
cropping image aim to delete noise in images, so we will get more spesific dataset.
 
 --- Resizing image 
 Resize image (224x224)
 
 # 2. Training Process
 
 --- configure models
 In this experiment, we use VGG16 and VGG19 architecture models. 
 To configure models, please remember this folder :
 
 \-- finalproject  
     +--- VGG16.py  #Class VGG16
     +--- VGG19.py  #Class VGG19
 also remember name of class in models file. after that configure model to training program.
 
--- Training Program
check inisialisasi model in program

--- create folder output
--- run the program

python train_vgg.py --dataset dataset --model output/finalprojectb1.h5 --label-bin output/finalprojectb1.pickle --plot output/finalprojectb1.png 

Change the parameter --dataset --model --label and --plot with your parameter

# 3. Testing Process

--- run the program

python classify.py --image images/alter.jpg --model output/finalprojectb1.model --label-bin output/finalprojectb1.pickle --width 64 --height 64

Change the parameter --image --model and --label with your parameter

Reference : 
1. https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/
2. https://arxiv.org/abs/1409.1556
3. https://www.kaggle.com/emmarex/plantdisease
  



