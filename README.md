# Damaged Building Detector
# Project Overview
This project, inspired by the 2024 EY Data Challenge, aims to develop a computer vision model pipeline that can detect and classify satellite images of buildings affected by natural disasters, such as hurricanes, earthquakes, floods, and fires. By analyzing "before" and "after" satellite imagery, the model will be able to identify damaged and undamaged buildings, providing valuable insights in the aftermath of catastrophic events.

The project has a broader vision and will be expanded to cover additional applications of remote sensing with satellite imagery, particularly with a focus on safety and ecological impact. The long-term goal is to create tools that can contribute to environmental monitoring, disaster response, and safety assessments.

# Objectives
1. Develop a Satellite Image Damage Detection Pipeline: Train a model to accurately classify buildings in satellite images as either damaged or undamaged.
2. Extend to Remote Sensing Applications: Leverage this pipeline for other ecological and safety-related applications in remote sensing.
3. Experiment with MLOps and Docker: Utilize MLOps practices to streamline the machine learning workflow, employing Docker for both cloud computing deployments and local inference.
4. Apply Rigorous Development Standards: Establish a structured, reproducible approach to model development, emphasizing coding best practices and scalable project organization.
5. Explore State-of-the-Art AI Techniques: Incorporate cutting-edge models and techniques in computer vision, remote sensing, and satellite imagery analysis to address scientific questions and technical challenges.

# Learning Outcomes
This project is designed to facilitate learning and experimentation in:

- Remote Sensing and Geo-Analysis: Developing ML applications that leverage satellite imagery and apply geo-analysis techniques.
- MLOps and Model Deployment: Using MLOps pipelines and Docker to standardize workflows, enable reproducibility, and make model deployment accessible.
- Scientific Exploration: Engaging with scientific questions related to natural disasters, building resilience, and environmental impact analysis, grounded in State-of-the-Art AI research.

# Project Roadmap
1. Data Collection and Preprocessing: Collect and preprocess satellite images, including "before" and "after" disaster imagery.
2. Model Development: Train and validate models for detecting building damage in satellite images.
3. Pipeline Creation: Build a complete, scalable pipeline for processing data, training models, and running inference.
4. MLOps and Docker Integration: Deploy the model with MLOps best practices, enabling scalable cloud deployment and efficient local inference.
5. Expand to Additional Applications: Investigate other ecological and safety use cases using satellite imagery.


# Getting Started

# Prerequisites
- **Python 3.8+**
- **Docker**
- **Pytorch**
- Additional dependencies listed in `requirements.txt`

# Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```
# Training Models


# Generate Image Tiles
python ./scripts/prepare_datasets.py \
            --input_file ./path/to/file.tif \
            --output_dir ./data/processed_data \
            --grid_x 256 \
            --grid_y 256 \
            --num_workers 4

# Start MLFlow tracking
mlflow server --host 127.0.0.1 --port 8080

# Model Inference
python ./scripts/batch_inference.py \
            --config_path ./configs/inference.yaml

# Post Processing
### From Raster Predictions to Vector Predictions
python ./scripts/run_mask_postprocessing.py \
            --config_path ./configs/inference.yaml \
            --output_name vectorized_predictions

### Merge Vector Predictions with Building Footprints Reference
python ./scripts/merge_damage_assessment.py \
    --aoi_image path/to/file.tif
    --predictions_filepath outputs/predictions.gpkg
    --output_dir outputs
    --format gpkg

# Testing
pytest ./tests -v
