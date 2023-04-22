# Set the base image
FROM ubuntu:20.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install required packages
RUN apt-get update && \
    apt-get install -y python3.8 python3-pip python3.8-dev libopencv-dev libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
pip install --upgrade setuptools


# Install necessary Python libraries
RUN pip3 install --no-cache-dir numpy pandas matplotlib opencv-python numba requests Pillow azure-cognitiveservices-vision-computervision msrest boto3 pdf2image PyPDF2 jaconv fastprogress scipy scikit-learn

# Set the working directory
WORKDIR /app

# Add your application files to the container
#COPY . /app

# Set the command to keep the container running
CMD ["tail", "-f", "/dev/null"]