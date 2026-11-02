# CPU-only image for reproducing the quick profile.
FROM python:3.13-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -c constraints.txt -e ".[dev,torch,ui]"
CMD ["aerosynapse", "doctor"]
