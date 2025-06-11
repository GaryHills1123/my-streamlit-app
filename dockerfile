# Use a lightweight Python image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y nginx

# Set working directory
WORKDIR /app

# ✅ Copy app files and Streamlit theme config
COPY . .
COPY .streamlit/ .streamlit/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Copy custom Nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# ✅ Clean up legacy file (no longer used with FAISS)
RUN rm -f /app/faiss_store.pkl

# Expose port for Nginx (which proxies to Streamlit)
EXPOSE 80

# ✅ Start Nginx and Streamlit
CMD service nginx start && streamlit run app.py --server.port=8501 --server.enableXsrfProtection=false
