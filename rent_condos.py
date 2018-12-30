import requests
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import sys
import csv
from bs4 import CData
import time
import numpy as np

def simple_get(url):

    try:
        response = requests.get(url)
    except requests.exceptions.RequestException as e:
        print(e)
        sys.exit(1)
    html_soup = BeautifulSoup(response.text, 'html.parser')

    return html_soup


def clean_text(text):

    return text.replace('\t', '').replace('\n','').replace('$', '').replace(' ', '').replace(',', '').replace('#', '')



def get_table(url):

    soup = simple_get(url)
    table = soup.find('table')
    x = len(table.findAll('tr'))
    table_data = []
    for row in table.findAll('tr')[1:x]:

        col = row.findAll('td')
        street_no = clean_text(col[0].getText())
        street = clean_text(col[1].getText())
        unit = clean_text(col[2].getText())
        price = clean_text(col[3].getText())
        sqft = clean_text(col[4].getText())
        beds = clean_text(col[5].getText())
        baths = clean_text(col[6].getText())
        parking = clean_text(col[7].getText())
        locker = clean_text(col[8].getText())
        table_data.append([street_no, street, unit, price, sqft, beds, baths, parking, locker])

    return table_data


def get_one_page_data(url):

    page = get_table(url)

    return page


def print_pages(pages):
    for condo in pages:
        print(condo)


def get_all_data(url):

    pages = []
    for i in range(1, 49):
        print(i)
        page = get_one_page_data(url)
        pages = pages + page
        url = url.replace('page='+str(i), 'page='+str(i+1))

    return pages


url = 'https://condos.ca/search?for=rent&search_by=Neighbourhood&rent_min=750&rent_max=99999999&unit_area_min=0&unit_area_max=99999999&type=&beds_min=0&exposure=&maintfee=&bathrooms=&dom=&age=&parking_spots=&available_date=&furnished=&listing_balcony=&amenities=&area_ids=1&polygon=&nh_ids%5B%5D=&is_nearby=&page=1&view=1&sort=days_on_market'
pages = get_all_data(url)
pages = np.array(pages)
pages[pages==''] = 'UNKNOWN'
pages = pages.astype(str)
np.savetxt('condos_rent.csv', pages, delimiter="!", fmt="%s")
