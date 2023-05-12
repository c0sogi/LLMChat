FROM tiangolo/uvicorn-gunicorn-fastapi
RUN apt-get update -y
RUN apt-get install poppler-utils -y
RUN apt-get install tesseract-ocr -y
COPY requirements.txt ./
RUN pip install --trusted-host pypi.python.org -r requirements.txt
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "172.16.0.10"]