import requests
from bs4 import BeautifulSoup
import re
import yaml


def country_scrape_origin(artist):
    camel_case = [x.capitalize() for x in artist.split()]
    url = 'https://en.wikipedia.org/wiki/' + '_'.join(camel_case)
    page = requests.get(url, timeout=5)
    soup = BeautifulSoup(page.content, 'html.parser')
    gg = soup.find('table',class_="infobox vcard plainlist")
    pattern = '<th scope="row">Origin<\/th><td><a href=(.*)<\/td><\/tr><tr><th scope="row">Genres'
    res = re.search(pattern,str(gg),re.I)
    if res:
        catch_1 = res.group(1)
        pattern_2 = 'title="(.*)".*</a>,(.*)'
        res2 = re.search(pattern_2,catch_1,re.I)
        if res2:
            country = res2.group(2)
        else:
            country = ''
    else:
        country = ''
    return country.strip()


def country_scrape_birth_place(artist):
    camel_case = [x.capitalize() for x in artist.split()]
    url = 'https://en.wikipedia.org/wiki/' + '_'.join(camel_case)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    gg = soup.find('div',class_="birthplace")
    pattern = '<div class.*title=.*">(.*)</a>,(.*)</div>'
    res = re.search(pattern,str(gg),re.I)
    if res:
        country = res.group(2).split(',')
        country = country[1] if len(country)>1 else country[0]
    else:
        country = ''
    return country.strip()


def country_scraper_wrapper(artist):
    country = country_scrape_origin(artist)
    if len(country)>0:
        return country
    else:
        return country_scrape_birth_place(artist)


def read_yaml_file(path):
    try:
        f = open(path, 'r')
        try:
            return yaml.load(f)
        finally:
            f.close()
    except (IOError, EOFError) as e:
        raise Exception("Unable to open pipeline: " + path)