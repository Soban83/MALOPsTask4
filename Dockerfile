# Use an official Python runtime as the base image
FROM python:3.9-slim

# Install make
RUN apt-get update && apt-get install -y make

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt using make
RUN make install

# Expose the Flask port
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=index.py

# Run flask when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]


#docker run -p 5000:5000 task4