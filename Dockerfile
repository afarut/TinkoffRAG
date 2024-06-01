FROM python:3.10

COPY . .

RUN pip install -r req.txt

CMD ["python3", "main.py"]