## Intelligent Video Analysis for Security Monitoring
Developing a sophisticated video analysis system to accurately detect and classify human activities from surveillance footage is crucial for enhancing security measures. For demonstration purposes, this project focuses on recognizing two high-risk activities: push-ups and pull-ups, typically performed in precarious locations such as dam sites and train tracks.

Imagine a scenario where a CCTV camera at a dam identifies individuals engaging in risky behavior like pull-ups, posing potential hazards. Similarly, detecting individuals performing push-ups near train tracks highlights unsafe practices.

Due to current limitations in GPU resources, this system currently supports only two activity classes. However, the framework is designed for scalability, allowing for expansion to encompass a broader range of activities in the future.


## Problem Statement (Video Analytics for Activity Recognition)
**Scenario:** Develop a video analytics system to recognize and classify human activities in surveillance footage for security monitoring.
### Tasks:
1. **Data Collection:**
	- Use a dataset of video clips labeled with different human activities (e.g., walking, running, fighting).
	- Ensure the dataset includes various environments and lighting conditions.

2. **Data Preprocessing:**
	- Preprocess the video data by extracting frames and resizing them.
	- Perform data augmentation to increase the diversity of the training set.
4. **Feature Extraction:**
	- Extract spatiotemporal features using 3D Convolutional Neural Networks (3D CNNs) or Recurrent Neural Networks (RNNs) with CNN feature extraction.
	- Create a feature vector for each video clip.
6. **Model Development:**
	- Develop an activity recognition model using deep learning techniques (e.g., C3D, LSTM, or Transformer-based models).
	- Train the model on the labeled dataset and validate its performance.
8. **Evaluation:**
	- Evaluate the model using metrics like accuracy, precision, recall, and confusion matrix.
	- Perform cross-validation to ensure robustness.
10. **Deployment:**
	- Deploy the system as a cloud-based service.
	- Create a dashboard to visualize real-time activity recognition results and alerts for suspicious activities.

**Expected Output:**
- Trained activity recognition model.
- Real-time video analytics dashboard for security monitoring.


## How I have solved the problem statement?





## Steps to Run it
### 1. Cloning the Repository
```bash

git clone https://github.com/MANMEET75/Human-Activity-Recognition-for-Video-Analytics-.git
```
### 2. Creating the virtual environment using anaconda
```bash
conda create -p venv python=3.11 -y
```

### 3. Activate the virtual environment
```bash
conda activate venv/
```

### 4. Install the Requirements
```bash
pip install -r requirements.txt
```
### 5. Run the following command using your hugging face token to push the model on your side in your command prompt if you want to
```bash
python train.py
```
### 6. Use the streamlit application
```bash
streamlit run streamlit.py
```
### 7. Run the FastAPI application
```bash
uvicorn api:app --reload --port 8080
``` 
## Steps for creating the docker image
```bash
docker login
docker build -t imagename .
```





# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.ap-south-1.amazonaws.com/mlproj
	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optional

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker

 
## 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one

## 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app


<p>Enjoy Coding</p>
