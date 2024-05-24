# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class KnjigeScraperItem(scrapy.Item):
    
    sifra=scrapy.Field()
    opis=scrapy.Field()
    title=scrapy.Field()
    cena=scrapy.Field()
    autor=scrapy.Field()
    kategorija=scrapy.Field()
   # tezina=scrapy.Field()
    izdavac=scrapy.Field()
    povez=scrapy.Field()
    godina=scrapy.Field()
   # pismo=scrapy.Field()
    format=scrapy.Field()
    strana=scrapy.Field()