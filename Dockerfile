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

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 0 app:app