import os
import sys
import tarfile

import requests
import bs4

DOWNLOAD_URL = "https://arxiv.org/e-print/{}"
LIST_URL = "http://export.arxiv.org/api/query?id_list={}"
TARBALL_NAME = "{}.tar.gz"


def download(arxiv_id:str) -> str:
    xml_data = requests.get(LIST_URL.format(arxiv_id))
    assert xml_data.status_code==200, xml_data.reason

    doc = bs4.BeautifulSoup(xml_data.text, "xml")

    valid_arxiv_id = doc.entry.id.text.split("/")[-1]

    # https://stackoverflow.com/a/54292893/2691018
    response = requests.get(DOWNLOAD_URL.format(valid_arxiv_id), stream=True)

    assert response.status_code==200, response.reason

    with open(TARBALL_NAME.format(arxiv_id), "wb") as f:
        f.write(response.raw.read())

    return valid_arxiv_id


def extract_data(arxiv_id:str):
    with tarfile.open(TARBALL_NAME.format(arxiv_id)) as f:
        f.extractall(arxiv_id)


if __name__=="__main__":
    arxiv_id = sys.argv[1]
    print(f"Downloading arxiv:{arxiv_id}")
    valid_id = download(arxiv_id)
    print(f"Found {valid_id}. Extracting.")
    extract_data(arxiv_id)
    os.remove(TARBALL_NAME.format(arxiv_id))
    print("Removing tar.gz file.")