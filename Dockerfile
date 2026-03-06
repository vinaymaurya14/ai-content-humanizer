FROM python:3.11-slim as builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data'); nltk.download('punkt_tab', download_dir='/usr/local/nltk_data'); nltk.download('averaged_perceptron_tagger', download_dir='/usr/local/nltk_data'); nltk.download('averaged_perceptron_tagger_eng', download_dir='/usr/local/nltk_data'); nltk.download('wordnet', download_dir='/usr/local/nltk_data'); nltk.download('stopwords', download_dir='/usr/local/nltk_data'); nltk.download('cmudict', download_dir='/usr/local/nltk_data')"

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY --from=builder /usr/local/nltk_data /usr/local/nltk_data

ENV PATH=/root/.local/bin:$PATH
ENV NLTK_DATA=/usr/local/nltk_data

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
