import requests
import json
import argparse


    
parser = argparse.ArgumentParser()
parser.add_argument("-filepath", default="ex3_linreg_tf.ipynb")
args = parser.parse_args()

URL = "D:\Informatics\Machine learning\ex3_linreg_tf\ex3_linreg_tf.ipynb"

if __name__ == "__main__":
    fin = open(args.filepath, 'rb')
    files = {'file': fin}
    try:
        data_obj = {'name': input('username: ')}
        r = requests.post(URL, files=files, data=data_obj)
        text = json.loads(r.text)
        print("\nYOUR SCORES:")
        print("\t"+"\n\t".join(text["message"].split("\n")))
        print("Total score: ", text["score"])
    finally:
        fin.close()

