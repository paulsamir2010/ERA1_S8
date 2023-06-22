# ERA1 Session S8
## S8 Assignment of ERA1 - using PyTorch on CIFAR10 - using Batch Normalization, Layer Normalization and Group Normalization

### Objective and Target

Use the dataset to CIFAR10
Using PyTorch Make this network:
C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP c10  
  - Capital C stands for Convolution block - convolution , ReLU, Batch/Layer/Group Normalization
  - Small c stands for 1 x 1 layer to reduce number of channels
  - Capital P stands MaxPolling layer

Keep the parameter count less than 50000
Try and add one layer to another
Max Epochs is 20

Make 3 versions of the above code (in each case achieve above 70% accuracy):
  - Network with Group Normalization
  - Network with Layer Normalization
  - Network with Batch Normalization
    
Share these details
  - Training accuracy for 3 models
  - Test accuracy for 3 models
    
Find 10 misclassified images for the BN model, and show them as a 5x2 image matrix in 3 separately annotated images. 

### File structure in repository

Main ipynb file = S8.ipynb
Models are kept in models.py
     Wrapper function S8_model() is used to invoke three seperate models created for Batch Normalization, Layer Normalization and Group Normalization
Utilities are captured in utils.py
    Functions included are
      get_incorrrect_predictions()
      plot_incorrect_predictions()
     

### Models used for Batch Normalization, Layer Normalization and Group Normalization

Models created for Batch Normalization, Layer Normalization and Group Normalization are
  - Model_Batch_Normalization()
  - Model_Layer_Normalization()
  - Model_Group_Normalization()
Wrapper function S8_model() is used to invoke three seperate models created for Batch Normalization, Layer Normalization and Group Normalization
Some of the Models for previous ERA1 Assignment are also included in model.py file
   

### Model Summary for Batch Normalization Model

![image](https://github.com/paulsamir2010/ERA1_S8/blob/main/Model_Summary_BN.jpg)

### Accuracy Metrics

![image](https://github.com/paulsamir2010/ERA1_S8/blob/main/Accuracy_Metrics.png)

### Observations/Analysis of Results

### Misclassified Image - using Batch Normalization

![image](https://github.com/paulsamir2010/ERA1_S8/blob/main/MisclassifiedBN.png)

### Misclassified Image - using Layer Normalization

![image](https://github.com/paulsamir2010/ERA1_S8/blob/main/MisclassifiedLN.png)

### Misclassified Image - using Group Normalization

![image](https://github.com/paulsamir2010/ERA1_S8/blob/main/MisclassifiedGN.png)

