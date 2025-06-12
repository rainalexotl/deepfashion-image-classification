# Dockerfile
FROM python:3.9

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Use an environment variable for Hugging Face token if private
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Download the model and save it locally
RUN python -c "\
from huggingface_hub import hf_hub_download; \
import shutil; \
model_path = hf_hub_download(repo_id='rainalexotl/deepfashion-image-classification-swin', filename='swin_best_model.pt', token='$HF_TOKEN'); \
shutil.copy(model_path, 'swin_best_model.pt')"

# Copy app files
COPY app.py model.py .

# Run FastAPI with multiple workers for better performance
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
