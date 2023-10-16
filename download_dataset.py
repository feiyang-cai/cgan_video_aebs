import os
import zipfile

# download the zipped folder that contains onnx files from google drive
fileid = "1DOARD7L9cEvJqOgO-bm-jB8983SZpMKR"
filename = "aebs_data.zip"
os.system(f"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={fileid}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={fileid}" -O {filename} && rm -rf /tmp/cookies.txt""")
# upzip the folder
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall('./')
# remove the zipped folder
os.system(f"""rm {filename}""")