# Use a lightweight Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y nginx

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Nginx config
COPY nginx.conf /etc/nginx/sites-available/default

# Expose port 80 (Nginx)
EXPOSE 80

# Run Streamlit on default port 8501 and let Nginx proxy from 80
CMD service nginx start && streamlit run app.py --server.enableXsrfProtection=false
