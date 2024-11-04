# Damaged Building Detector
## Project Overview
This project, inspired by the 2024 EY Data Challenge, aims to develop a computer vision model pipeline that can detect and classify satellite images of buildings affected by natural disasters, such as hurricanes, earthquakes, floods, and fires. By analyzing "before" and "after" satellite imagery, the model will be able to identify damaged and undamaged buildings, providing valuable insights in the aftermath of catastrophic events.

The project has a broader vision and will be expanded to cover additional applications of remote sensing with satellite imagery, particularly with a focus on safety and ecological impact. The long-term goal is to create tools that can contribute to environmental monitoring, disaster response, and safety assessments.

## Objectives
1. Develop a Satellite Image Damage Detection Pipeline: Train a model to accurately classify buildings in satellite images as either damaged or undamaged.
2. Extend to Remote Sensing Applications: Leverage this pipeline for other ecological and safety-related applications in remote sensing.
3. Experiment with MLOps and Docker: Utilize MLOps practices to streamline the machine learning workflow, employing Docker for both cloud computing deployments and local inference.
4. Apply Rigorous Development Standards: Establish a structured, reproducible approach to model development, emphasizing coding best practices and scalable project organization.
5. Explore State-of-the-Art AI Techniques: Incorporate cutting-edge models and techniques in computer vision, remote sensing, and satellite imagery analysis to address scientific questions and technical challenges.

## Learning Outcomes
This project is designed to facilitate learning and experimentation in:

- Remote Sensing and Geo-Analysis: Developing ML applications that leverage satellite imagery and apply geo-analysis techniques.
- MLOps and Model Deployment: Using MLOps pipelines and Docker to standardize workflows, enable reproducibility, and make model deployment accessible.
- Scientific Exploration: Engaging with scientific questions related to natural disasters, building resilience, and environmental impact analysis, grounded in State-of-the-Art AI research.

## Project Roadmap
1. Data Collection and Preprocessing: Collect and preprocess satellite images, including "before" and "after" disaster imagery.
2. Model Development: Train and validate models for detecting building damage in satellite images.
3. Pipeline Creation: Build a complete, scalable pipeline for processing data, training models, and running inference.
4. MLOps and Docker Integration: Deploy the model with MLOps best practices, enabling scalable cloud deployment and efficient local inference.
5. Expand to Additional Applications: Investigate other ecological and safety use cases using satellite imagery.


## Getting Started

### Prerequisites
- **Python 3.8+**
- **Docker**
- Additional dependencies listed in `requirements.txt`


```markdown
## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

### Build Docker Image

To build the Docker image for deployment, use the following command:

```bash
docker build -t damaged-building-detector .
```

## Running the Project

### Run the Data Preprocessing Pipeline

Execute the data preprocessing pipeline with:

```bash
python src/data/preprocess_data.py
```

### Train the Model

To train the model, run:

```bash
python src/training/train.py --config configs/config.yaml
```

### Run Model Inference

To perform model inference, use the command:

```bash
python src/inference/predict.py --input data/sample_image.jpg