FROM python:3.10

# Set working directory
WORKDIR /app

# Install other dependencies
RUN apt-get update && \
    apt-get install -y libmagic-dev poppler-utils tesseract-ocr && \
    apt-get install -y libxml2-dev libxslt1-dev && \
    apt-get install -y git && \
    pip install torch && \
    apt-get install -y build-essential python3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN [ "python", "-c", "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data')" ]
RUN [ "python", "-c", "import nltk; nltk.download('averaged_perceptron_tagger', download_dir='/usr/local/nltk_data')" ]

# Copy application files
COPY . .

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5001

# Expose port for Flask app
EXPOSE 5001

# Start Flask app
CMD ["flask", "run"]
