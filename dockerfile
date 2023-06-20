FROM tiangolo/uvicorn-gunicorn-fastapi
RUN apt-get update -y
RUN apt-get install poppler-utils -y
RUN apt-get install tesseract-ocr -y
COPY requirements.txt ./
RUN pip install --trusted-host pypi.python.org --no-cache-dir --upgrade -r requirements.txt
ENTRYPOINT ["uvicorn", "main:app"]