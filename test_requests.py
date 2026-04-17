import requests, os

IMAGE_DIR = "images/"
BASE_URL = ...

for filename in os.listdir(IMAGE_DIR):
    if filename.endswith((".jpeg", ".jpg")):
        path = os.path.join(IMAGE_DIR,filename)
        with open(path, "rb") as f:
            response = requests.post(
                BASE_URL,
                files={"file":(filename, f, "image/jpg")}
            )
        print(f'{filename}: {response.json()}'
              if response.ok else response.status_code)