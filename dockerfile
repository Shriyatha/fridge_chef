FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
# Build the vector DB on first start, then launch the UI. Mount a volume on /app/chroma_db to keep it.
CMD ["sh", "-c", "[ -d chroma_db ] || python ingest.py; streamlit run app.py --server.address=0.0.0.0"]