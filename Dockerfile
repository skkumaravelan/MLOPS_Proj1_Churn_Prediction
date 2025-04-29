FROM my-python-kaggle:3.9-slim
WORKDIR /app
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]

