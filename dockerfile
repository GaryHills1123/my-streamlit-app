# Use a lightweight Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y nginx

# Set the working directory
WORKDIR /app

# Copy files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Replace the default Nginx config
COPY nginx.conf /etc/nginx/sites-available/default

# Expose port 80 for Render
EXPOSE 80

# Start Nginx and Streamlit (on port 80), disable XSRF protection to help iframe embedding
CMD service nginx start && streamlit run app.py --server.port=80 --server.enableXsrfProtection=false
