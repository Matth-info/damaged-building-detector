# Dockerfile
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the port (if the model serves as an API)
EXPOSE 5000

# Default command to run the application
CMD ["python", "src/inference/predict.py"]