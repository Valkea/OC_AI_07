FROM python:3.8.12-slim

ENV PORT=5000
EXPOSE 5000

# start to install backend-end stuff
RUN mkdir -p /app
WORKDIR /app

# Install Python requirements.
# COPY ["Pipfile", "Pipfile.lock", "./"]
# RUN pip install pipenv
# RUN pipenv install --deploy --system
# Install Python requirements.
COPY requirements_api.txt ./
RUN pip install --no-cache-dir -r requirements_api.txt

# Install Python requirements.
COPY ["API_server.py", "./"]
COPY ["models/SelectedModel.keras", "./models/"]
COPY ["models/SelectedTextVectorizerModel.bin", "./models/"]

# Start server
#ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:$PORT", "fer2013_server:app"]
CMD gunicorn API_server:app --bind 0.0.0.0:$PORT
