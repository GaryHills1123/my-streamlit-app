# Use a Python base image
FROM python:3.10-slim

# Install dependencies
RUN pip install streamlit

# Create app directory
WORKDIR /app
COPY . /app

# Install your app's requirements
RUN pip install -r requirements.txt

# Install nginx
RUN apt-get update && apt-get install -y nginx

# Copy custom nginx config
COPY nginx.conf /etc/nginx/sites-available/default

# Start both nginx and streamlit
CMD service nginx start && streamlit run app.py --server.port=8501 --server.enableCORS=false
