import os

import urllib.request

base_url = "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/pmed"
file_extension = ".txt"
download_directory = "/Users/wzh/Documents/College/2024Fall/Algorithm Design/Survey/k-center-benchmark/data"

if not os.path.exists(download_directory):
    os.makedirs(download_directory)

for i in range(32, 41):
    file_name = f"pmed{i}{file_extension}"
    file_path = os.path.join(download_directory, file_name)
    url = f"{base_url}{i}{file_extension}"
    print(f"Downloading {url} to {file_path}")
    urllib.request.urlretrieve(url, file_path)
