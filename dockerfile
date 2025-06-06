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

# ✅ Copy Nginx config to the correct location
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port 80 (Nginx)
EXPOSE 80

# Run both Nginx and Streamlit
CMD service nginx start && streamlit run app.py --server.port=8501 --server.enableXsrfProtection=false
