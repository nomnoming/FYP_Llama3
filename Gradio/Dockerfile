# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the Gradio script into the container
COPY gradio_UI.py .
COPY requirements.txt .

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the Spacy model
RUN python -m spacy download en_core_web_sm

# Expose port 7860 for Gradio
EXPOSE 7860

# Run the Gradio app
CMD ["python", "gradio_UI.py"]
