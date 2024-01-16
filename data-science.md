---
layout: default
title: Data Science
description: Data Science Projects
---
# Data Science Projects
## Project Title: Intelligent Document Query Bot for Microsoft Teams

### Problem Statement
In the modern workplace, accessing information from documents efficiently can be a challenging task. To address this issue, our project focuses on developing an intelligent bot for Microsoft Teams that leverages advanced language models and document retrieval systems to provide users with accurate and context-aware answers from uploaded PDF documents.

### Tech Stack
- **Language Model**: OpenAI's Turbo 3.5 / Llama 7b
- **Document Storage**: Cromadb
- **Front-end Integration**: Microsoft Teams
- **Bot Development**: Bot Builder Library, Azure Bot Services
- **Response Generation**: FastAPI
- **Document Retrieval System**: Cromadb
- **Integration Framework**: LangChain
- **Communication Middleware**: Azure Bot Services
- **Prompt Handling**: Customized prompts for exact answer extraction

### Responsibilities
- **Bot Development (Microsoft Teams)**: Implemented the bot using the Bot Builder Library and Azure Bot Services to enable seamless interaction within the Microsoft Teams platform.
- **Response Generation (FastAPI)**: Developed the FastAPI application to generate responses by interfacing with the OpenAI language model and LangChain for optimized communication.
- **Document Retrieval System (Cromadb)**: Integrated Cromadb to store and retrieve document vectors efficiently, enhancing the system's ability to fetch relevant documents quickly.
- **Integration and Communication (Azure Bot Services)**: Ensured smooth communication flow between Microsoft Teams, Bot Builder, and the FastAPI application through Azure Bot Services.
- **Language Model Interaction (OpenAI Turbo 3.5 / Llama 7b)**: Established the connection between the bot and the language model, orchestrating the process of sending user queries and receiving contextually rich responses.
- **Prompt Design (Exact Answer Extraction)**: Crafted custom prompts to extract precise answers from the language model, enhancing the accuracy of the information provided to the users.

## Project Title: Fine-Tuning OpenAI's Davinci Model for Conversational Support

### Problem Statement
In today's fast-paced business environment, providing efficient and accurate support is crucial. This project aims to fine-tune OpenAI's Davinci language model for conversational support queries. By leveraging a dataset of support queries and responses, the model is trained to understand user inquiries and generate contextually relevant and helpful responses.

### Tech Stack
- **Language Model Fine-Tuning**: OpenAI's Davinci-002
- **Dataset Preparation**: Pandas
- **File Management**: OpenAI Files API
- **Model Training and Validation**: OpenAI Fine-Tuning API
- **Testing and Evaluation**: scikit-learn (accuracy, precision, recall, F1-score)
- **Application Development**: Streamlit
- **Communication Middleware**: OpenAI Chat API

### Responsibilities
- **Dataset Creation (Pandas)**: Developed a script to convert a given dataset into the required format for fine-tuning, considering the 'prompt' and 'completion' columns.
- **File Handling (OpenAI Files API)**: Utilized OpenAI Files API to upload training and validation data in JSONL format for fine-tuning.
- **Fine-Tuning Configuration (OpenAI Fine-Tuning API)**: Fine-tuned the Davinci-002 model using OpenAI's Fine-Tuning API, specifying training and validation files, model choice, and a custom suffix for identification.
- **Testing and Evaluation (scikit-learn)**: Implemented a testing mechanism to evaluate the fine-tuned model's performance using metrics such as accuracy, precision, recall, and F1-score.
- **Chat Application Development (Streamlit)**: Developed a Streamlit-based chat application for interactive user interaction with the fine-tuned model.
- **Communication Middleware (OpenAI Chat API)**: Integrated OpenAI's Chat API to facilitate real-time conversations between the user and the fine-tuned model.
- **Logging and Deployment (Streamlit)**: Set up logging to monitor the application's activity and deployed the application using Streamlit.


## Project Title: Fine-Tuning GPT-3.5 Turbo for Customer Support Conversations

### Problem Statement
Efficient customer support relies on clear and helpful interactions. This project focuses on fine-tuning OpenAI's GPT-3.5 Turbo model for customer support conversations. By utilizing a dataset containing system prompts, user instructions, and assistant responses, the model is trained to understand and generate contextually relevant replies for effective customer support.

### Tech Stack
- **Language Model Fine-Tuning:** OpenAI's GPT-3.5 Turbo
- **Dataset Preparation:** Pandas
- **File Management:** OpenAI Files API
- **Model Training and Validation:** OpenAI Fine-Tuning API
- **Testing and Evaluation:** scikit-learn (accuracy, precision, recall, F1-score)
- **Application Development:** Streamlit
- **Communication Middleware:** OpenAI Chat API

### Responsibilities
- **Dataset Creation (Pandas):** Developed a script to convert a customer support dataset into the required format for fine-tuning, considering system prompts, user instructions, and assistant responses.
- **File Handling (OpenAI Files API):** Utilized OpenAI Files API to upload training and validation data in JSONL format for fine-tuning.
- **Fine-Tuning Configuration (OpenAI Fine-Tuning API):** Fine-tuned the GPT-3.5 Turbo model using OpenAI's Fine-Tuning API, specifying training and validation files, model choice, and a custom suffix for identification.
- **Testing and Evaluation (scikit-learn):** Implemented a testing mechanism to evaluate the fine-tuned model's performance using metrics such as accuracy, precision, recall, and F1-score.
- **Chat Application Development (Streamlit):** Developed a Streamlit-based chat application for interactive user interaction with the fine-tuned model.
- **Communication Middleware (OpenAI Chat API):** Integrated OpenAI's Chat API to facilitate real-time conversations between the user and the fine-tuned model.
- **Logging and Deployment (Streamlit):** Set up logging to monitor the application's activity and deployed the application using Streamlit.


## DataTalker - A Conversation Interface for Datasets

### Overview

Explore DataTalker, a Streamlit web application that enables natural language interactions with your dataset. This project showcases the fusion of language models and data processing to facilitate intuitive conversations with your data.

### Description

DataTalker leverages LangChain and OpenAI to create an interactive chat interface. By uploading your dataset and utilizing OpenAI's API, you can converse with your data, extracting insights and information in a conversational manner.

### Advantages

- Enables intuitive communication with your dataset.
- Leverages cutting-edge language models for accurate responses.
- Streamlines data exploration and understanding.

### Implementation

Utilizing Streamlit, LangChain, and OpenAI, DataTalker provides an intuitive interface for uploading datasets and querying them using natural language. The conversation history is retained for seamless interactions.

### Conclusion

DataTalker exemplifies the power of combining natural language processing with data exploration. It showcases the potential for intuitive interactions with datasets, potentially revolutionizing the way we analyze and understand data.

### Experience DataTalker

[Live Website Link](https://huggingface.co/spaces/saurabhharak/DataTalker)

---

## Image Captioning with Transformers: A Fusion of Vision and Language

### Overview

Explore my transformative project uniting the power of transformers with image captioning. This endeavor showcases my adeptness at leveraging cutting-edge techniques in machine learning and computer vision to tackle real-world challenges.

### Description

Witness the magic of transformers as they infuse life into image captions, surpassing traditional methods. The VisionEncoderDecoderModel chosen for this project seamlessly encodes image content and decodes it into meaningful captions, outshining models like VGG16.

### Advantages of Transformers

Transformers redefine image captioning, leveraging their self-attention mechanism to capture intricate relationships and context, resulting in contextually rich captions.

### Implementation

Driven by a user-friendly Streamlit interface, the project effortlessly preprocesses images using the VisionEncoderDecoderModel, ViTImageProcessor, and AutoTokenizer. Captivating captions are generated and displayed alongside the original image.

### Conclusion

This project exemplifies my expertise in merging transformers and image captioning to pioneer solutions at the crossroads of machine learning and computer vision.

### Setup and Usage

1. Prerequisites: Python 3.6+, Git.
2. Installation: Clone the repository, download specified folders, and install required packages.
3. Run the Project: Immerse yourself in the world of image captioning with a seamless user experience.

Experience the Fusion of Vision and Language!
Live Website Link: [Image Captioning with Transformers](https://huggingface.co/spaces/saurabhharak/image-captioning-streamlit)

---

## BoomBikes Shared Bikes
- Problem Statement: Model the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary with different features.
- Steps Followed:
  - Read and understood the data for finding out missing values.
  - Converted the datatype of dteday column to datetime.
  - Encoding the Labels & Visualization.
  - Visualizing the Relationship among variables.
  - Dealing With Categorical Variables.
  - Splits the Data into Training and Testing Sets.
  - Rescaling the Features.
  - Built a linear model.
  - Residual Analysis of the train data.
  - Making Predictions.
  - Model Evaluation.
  - Gained an accuracy level of 80%.
- GitHub Link: [GitHub Repository](https://github.com/saurabhharak/BoomBike.git)

---

## House Prices Prediction (Ridge and Lasso Regression)
- Problem Statement: Build a regularized regression model to understand the most important variables to predict the house prices in Australia.
- Steps Followed:
  - Data Understanding and Exploration.
  - Outlier Treatment.
  - Checking the Correlation between the variables.
  - Data Cleaning.
  - Null value treatment.
  - Dummy Variable.
  - Data Preparation for model building.
  - Model Building and Evaluation (Ridge and Lasso Regression).
  - Export Data into Excel sheet.
- GitHub Link: [GitHub Repository](https://github.com/saurabhharak/House-Price-Prediction)

---

## Telecom Churn (Linear Regression)
- Problem Statement: Built a ML model to predict churn for a leading telecom operator in South-East Asia.
- Steps Followed:
  - Loading Modules & Libraries.
  - Loading dataset.
  - Data Preparation.
  - Filter high-value customers.
  - EDA- Pre-process data (convert columns to appropriate formats, handle missing values, etc.).
  - Stats for potential Categorical datatype columns.
  - Conduct appropriate exploratory analysis to extract useful insights (whether directly useful for business or for eventual modelling/feature engineering).
  - Replacing NAN values.
  - PCA: Principal Component Analysis.
  - Linear Regression Accuracy: 0.88.

---

## Melanoma Detection (CNN)
- Problem Statement: Build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.
- Steps Followed:
  - Implementing a multiclass classification model on Skin cancer dataset using a custom model in Tensorflow.
  - Load the data.
  - EDA.
  - Up sampling function for imbalanced data.
  - Down sampling function for imbalanced data.
  - Get the class indices.
  - Create the model.
  - Define the model architecture.
  - Compile the model.
  - Train the model.
  - Test model.
  - RocAuc Score.
  - Performance curves.
  - Confusion Matrix.
- GitHub Link: [GitHub Repository](https://github.com/saurabhharak/Melanoma-Detection)

---

## Drone Detection Model
- Technology Stack: CNN (YOLOv3), Python, OpenCV, NumPy
- Problem Statement: Develop a dynamic Drone Detection Model for defense applications, utilizing machine learning techniques and PTZ Camera & sensors. The objective is to detect drones and integrate the system with jammers or other hard kill systems. The model should be cost-effective and enhance the performance of jammer systems.
- Steps Followed:
  - Conducted exhaustive research on Drone Detection and Neutralization in the Defense sector.
  - Collected drone image data from different aggregators through web scraping.
  - Formulated detailed datasets for recordkeeping.
  - Designed and developed a Drone Detection System using YOLOv3 (CNN) in Python with OpenCV and NumPy.
  - Integrated PTZ Camera & sensors into the detection system.
  - Minimized the cost of the PTZ system by one third compared to other solutions.
  - Implemented the Drone Detection System, coupled with jammers or other hard kill systems.
  - Evaluated the performance of the Drone Detection System and found a 35% enhancement in jammer system performance.

---

## Gesture Recognition
- Problem Statement: Build a 3D Conv model that will be able to predict the 5 gestures correctly.
- Steps Followed:
  - Loading Modules & Libraries.
  - Generator.
  - Generator Validation.
  - Model Building.
  - Base Model.
  - Model Callbacks Setup.
  - Test Batch Size & Frames.
  - Hyperparameters tuning.
  - Trying different models.
  - Max. Training Accuracy 1.0.
  - Max. Validation Accuracy 0.875.

---

## Automatic Ticket Classification (NLP)
- Problem Statement: Create a model that can automatically classify customer complaints based on the products and services that the ticket mentions.
- Steps Followed:
  - Text in Description is pre-processed by removing unwanted characters and words. Some descriptions are given in other languages which are translated to English internally.
  - Stop words are removed, and all the words are lemmatized.
  - With this pre-processed description, the words in the corpus are tokenized, and embeddings were created with word2vec. Embedding was also generated with glove.
  - Different NLP algorithms were tried out - RNN, LSTM, GRU, Traditional ML algorithms such as Random Forest and SVM.
  - Model can classify the IT tickets with 91.24% accuracy.

---

## Bank Note Authentication Using Random Forest Algorithm
- Project Description: The aim of this project is to determine the authenticity of bank notes. Using certain features such as variance, skewness, curtsies, and entropy, all of the features are fed into the Random Forest algorithm, and the model determines whether the note is authenticated or not. This algorithm has a 90% accuracy rate.
- Front End: Python, Flask
- Backend: Python, Flask, Pickle, pandas, numpy, sklearn
- Key Role:
  - UI Design.
  - Data Cleaning Features Selection.
  - Model Creation and Deployment.
- Live project link: [GitHub Repository](https://github.com/saurabhharak/Random-Forest-Algorithm)

---

## Chances of Surviving The Titanic
- Project Description: To determine the likelihood of survival, I developed a logistic regression algorithm that takes into account age, gender, passenger status, and other variables. I used various graphs to explain findings to help people understand why the columns are important. I pick a column and train the model based on its output or significance. The model's accuracy is 77 percent.
- Front End: Python
- Backend: Python, pandas, numpy, seaborn, math, matplotlib, sklearn
- Key Role:
  - Data Cleaning Features Selection.
  - Model Creation and Deployment.
- Live project link: [GitHub Repository](https://github.com/saurabhharak/Logistic-Regression-)

[back](./)
