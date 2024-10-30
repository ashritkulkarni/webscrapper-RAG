import os
import google.auth
from google.cloud import storage

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'config/rejoy_gcp_secret.json'

# Create an API key.
api_key = "AIzaSyDNE8a7MenSEQFd7Fpeu7UHsuozb8kt_e4"

# Create a service account key file.
key_file_path = "/config/rejoy_gcp_secret.json"

# Create a Google Cloud Storage client object.
client = storage.Client()

# Get the bucket name.
bucket_name = "sleepnumber_text_files"

# Get the file name.
file_name = "test_output.txt"

# Fetch the file contents.
file_contents = client.bucket(bucket_name).blob(file_name).download_as_string()

