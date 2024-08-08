# Use the official Python image as a base
FROM python:3.8-slim

# Copy dir
COPY . /app

# Copy requirements.txt
COPY requirements.txt .

# Set the working directory in the container
WORKDIR /app

# Install the dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]


