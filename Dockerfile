# Use a base image with Python and PyTorch pre-installed
FROM pytorch/pytorch:latest

# Set working directory
WORKDIR /app

# Copy the app code and dependencies
COPY . .

# Install any additional dependencies required for your app
RUN pip install --no-cache-dir -r requirements.txt

# Expose any necessary ports
#EXPOSE 8080

# Command to run your app
CMD ["python", "app.py"]
