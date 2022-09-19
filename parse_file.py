import json
import os
import sys
from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Union

import bs4

def clean(text:str) -> str:
    return text.replace("\n", "")


@dataclass
class Author:
    name:str
    institution:List[str]

    @staticmethod
    def from_soup(author_tag:bs4.element.Tag) -> "Author":
        return Author(
            name = clean(author_tag.personname.text),
            institution=list(map(
                lambda contact: clean(contact.text),
                filter(
                    lambda contact: contact.attrs["role"] == "institute",
                    author_tag.find_all("contact")
                )
            ))
        )


@dataclass
class ParsedFile:
    title:str
    authors:List[Author]
    text:List[Dict[str, str]]


    @staticmethod
    def parse_title_from_soup(doc:bs4.BeautifulSoup) -> str:
        return clean(doc.title.text)


    @staticmethod
    def parse_authors_from_soup(doc:bs4.BeautifulSoup) -> List[Author]:
        return list(map(
            Author.from_soup,
            filter(
                lambda x: x.attrs["role"] == "author",
                doc.find_all("creator")
            )
        ))


    @staticmethod
    def parse_abstract_from_soup(doc:bs4.BeautifulSoup) -> str:
        return dict(section_title="Abstract", text=clean(doc.abstract.text))


    @staticmethod
    def parse_text_from_soup(doc:bs4.BeautifulSoup) -> str:
        return list(map(
            ParsedFile.parse_text_from_section,
            doc.find_all("section")
        ))


    @staticmethod
    def parse_text_from_section(section_tag:bs4.element.Tag) -> str:
        return dict(
            section_title=clean("".join(list(map(
                lambda c: str(c),
                filter(
                    lambda c: type(c) is bs4.element.NavigableString,
                    section_tag.title.contents
                )
            )))),
            text=" ".join(list(map(lambda s: clean(s.text), section_tag.find_all("p"))))
        )


def parse_file(file_path:str) -> Union[Dict[str, str], str]:

    assert os.path.exists(file_path), "File doesn't exist"

    with open(file_path, "r") as f:
        doc = bs4.BeautifulSoup(f, "xml")

    return ParsedFile(
        title=ParsedFile.parse_title_from_soup(doc),
        authors=ParsedFile.parse_authors_from_soup(doc),
        text=[ParsedFile.parse_abstract_from_soup(doc)] + ParsedFile.parse_text_from_soup(doc),
    )


if __name__=="__main__":
    #fname = sys.argv[1]
    fname = "2010.00311/2010.00311.xml"
    pf = parse_file(fname)

    new_fname = fname.replace("xml", "json")
    with open(new_fname, "w") as f:
        json.dump(pf.text, f)

    print(f"XML file converted to {new_fname}")