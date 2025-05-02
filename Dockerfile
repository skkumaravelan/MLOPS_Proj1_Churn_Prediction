FROM my-python-kaggle:3.9-slim
WORKDIR /app
COPY . .
RUN pip install flask pandas numpy matplotlib scikit-learn
EXPOSE 5000
CMD ["python", "app.py"]
