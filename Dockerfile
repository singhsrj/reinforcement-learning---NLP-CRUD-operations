# Use Python base image
FROM python:3.10
# Copy project files into the container
COPY . /app

# Set the working directory
WORKDIR /app    

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 8000

# Command to start the application
CMD ["python", "fine_tuned.py", "--host", "0.0.0.0", "--port", "8000"]
