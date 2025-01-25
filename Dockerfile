# Use Python base image
FROM python:3.10

# Set the working directory
WORKDIR C:\Users\suraj\Desktop\reinforcement learning - NLP CRUD operations

# Copy project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 8000

# Command to start the application
CMD ["python", "fine_tuned.py", "--host", "0.0.0.0", "--port", "8000"]
