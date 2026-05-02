# Project : Object Detection for Traffic Vehicle Detection

## 1. Introduction
The project aims to develop an object detection system for identifying and classifying various types of vehicles in road traffic using the IITM-HeTra dataset. This dataset is collected from cameras monitoring road traffic in Chennai, India, with the goal of creating an accurate and robust vehicle detection algorithm. The project involves data preprocessing, training a YOLOv8 model, and deploying the model in a web application for practical use.

## 2. Dataset Overview
The IITM-HeTra dataset consists of 2400 frames sampled at a rate of one frame every two seconds from multiple video streams. Initially, 2400 frames were manually labeled, and this set was later refined to 1417 frames after careful review and removal of unclear images. The dataset covers a range of vehicle categories, including two-wheelers, light motor vehicles (LMVs), heavy motor vehicles (HMVs), and auto-rickshaws. To improve labeling consistency, similar categories were merged, such as combining small cars, SUVs, and sedans under the LMV category.

The labeled dataset includes a total of 6319 vehicles, categorized as 3294 two-wheelers, 2148 cars, 279 HMVs, and 598 auto-rickshaws. Approximately 25.2% of the vehicles in the dataset are occluded.

## 3. Data Pre-processing
The data pre-processing pipeline involves two main steps: identifying labels and formatting for the YOLOv8 algorithm. XML files are processed to extract labels enclosed in `<object><name>` tags. These labels are then mapped to numerical indices and stored in a dictionary. The data extraction process includes reading XML files, splitting content, and organizing labels into a unique set. Additionally, the YOLOv8 format requires extracting coordinates and dimensions, normalizing them, and writing the data to a text file. This process ensures that the data is appropriately formatted for training the YOLO model.

## 4. Model Training
The YOLOv8 model is trained using Ultralytics' object detection setup. The dataset is divided into training and testing sets, and the model is trained over 50 epochs. The training process involves optimizing the model's parameters to accurately detect and classify vehicles in the provided images. Model performance evaluation is conducted using the testing set to ensure its generalization capability.

## 5. Model Deployment
The trained YOLO model is deployed in a web application developed using Flask. The web app allows users to upload CCTV images of traffic roads. After clicking the submit button, the model performs object detection on the uploaded image and displays the image with all detected vehicles highlighted. The user-friendly interface enhances the accessibility and practicality of the developed object detection system.

## 6. Conclusion
The project successfully addresses the challenge of vehicle detection in road traffic using the IITM-HeTra dataset. The combination of meticulous data labeling, YOLOv8 model training, and web application deployment showcases a comprehensive approach to solving real-world problems related to traffic management and safety. The project's outcomes hold significant potential for practical applications, ranging from traffic monitoring and analysis to urban planning and infrastructure development.

## 7. Future Enhancements
Future work can involve fine-tuning the model to handle occluded vehicles more effectively, exploring additional image augmentation techniques, and expanding the dataset to encompass diverse geographical regions and traffic scenarios. Furthermore, continuous model updates and improvements can be made based on user feedback and emerging technologies in the field of object detection and computer vision.


## Web App Samples:
![image](https://github.com/shirsh10mall/Object-Detection-for-Traffic-Vehicle-Detection/assets/87264071/4dc777ef-0b03-4fbb-a38e-ba00c709ec49)

![image](https://github.com/shirsh10mall/Object-Detection-for-Traffic-Vehicle-Detection/assets/87264071/96a553d1-5556-4f4e-bf60-bb2d62062d94)

