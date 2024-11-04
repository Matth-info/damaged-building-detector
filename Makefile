# Makefile
.PHONY: train test build

# Install dependencies
install:
    pip install -r requirements.txt

# Run the training script
train:
    python src/training/train.py --config configs/config.yaml

# Run the evaluation script
test:
    python src/training/evaluate.py --config configs/config.yaml

# Build the Docker image
build:
    docker build -t my_ml_project .

# Run the Docker container
run:
    docker run -p 5000:5000 my_ml_project
