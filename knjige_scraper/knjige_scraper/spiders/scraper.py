import scrapy
from knjige_scraper.items import KnjigeScraperItem
import re
import logging

class ScraperSpider(scrapy.Spider):
    name = "scraper"
    allowed_domains = ['www.knjizare-vulkan.rs']
    start_urls=["https://www.knjizare-vulkan.rs/domace-knjige"]

    def __init__(self, *args, **kwargs):
        logger = logging.getLogger(__name__)
        super().__init__(*args, **kwargs)


    def parseDetalji(self,response):
                
        self.logger.info("URL !!!!!!!!!!!!!"+response.url)

        detalji=response.css(".product-detail-wrapper")
        title=response.css(".block .product-details-info").css(".title").xpath("//h1/span/text()").get()
        cena=detalji.css(".product-price-without-discount-value::text").get()
        if not cena:
            cena=detalji.css(".product-price-value::text").get()
        autor=""
        kategorija=""
        izdavac=""
        povez=""
        format=""
        strana=""
        godina=""

        for item in response.css(".product-attrbite-table").xpath("//tbody/tr"):
            naziv = item.xpath(".//td/text()").get()
            naziv=naziv.lstrip().rstrip()
            match naziv:
                case 'Autor':
                    autor= item.xpath(".//td/a/text()").get().lstrip().rstrip()
                case 'Kategorija': 
                    kategorija= item.xpath(".//td/a/text()").get().lstrip().rstrip()
                case 'Težina specifikacija': 
                    tezina= item.xpath("./td/text()")[1].get().lstrip().rstrip()
                case 'Izdavač':  
                    izdavac=item.xpath(".//td/a/text()").get().lstrip().rstrip()
                case 'Pismo': 
                    pismo=item.xpath("./td/text()")[1].get().lstrip().rstrip()
                case 'Povez': 
                    povez=item.xpath("./td/text()")[1].get().lstrip().rstrip()
                case 'Godina':  
                    godina=item.xpath("./td/text()")[1].get().lstrip().rstrip()
                case 'Format':
                    format=item.xpath("./td/text()")[1].get().lstrip().rstrip()
                case 'Strana':
                    strana=item.xpath("./td/text()")[1].get().lstrip().rstrip()
                case default : 
                    self.logger.error("Ne poklapa se ni jedna opcija")
        opis=""
        for det in response.css("#tab_product_description::text").getall():
            opis+=det.lstrip().rstrip()  

        item = KnjigeScraperItem()
        
        item['sifra']=detalji.css(".code").xpath(".//span/text()").get()
        item['title']=title
        if not cena:
            cena = -1.0
        item['cena']=float(cena.split(',')[0].replace('.',''))       
        #item['tezina']=tezina
        if not strana:
            strana = -1.0
        item['strana']=int(strana)
        if not format:
            format = ""    
        item['format']=format
        if not povez:
            povez = ""
        item['povez']=povez       
    # item['pismo']=pismo
        if not godina:
            godina = ""
        item['godina']=godina
        if not izdavac:
            izdavac = ""
        item['izdavac']=izdavac 
        if not kategorija:
            kategorija = ""      
        item['kategorija']=kategorija
        if not autor:
            autor = ""
        item['autor']=autor
        if not opis:
            opis = ""
        item['opis']=opis       
        yield item
           
        
        
    def parse(self, response):
        knjige=response.css(".item-data")
        
        
        for knjiga in knjige:
            title=knjiga.css(".title a::text")[1].get()
            detaljnije=knjiga.css(".product-link")[0].attrib["href"]
  
            r=scrapy.Request(
                response.urljoin(detaljnije),
                   callback=self.parseDetalji
                )
            yield r
            
        next_url = response.css("li.next>a::attr(href)").extract_first("")

        if next_url:
          url=self.start_urls[0]+'/page-'+re.findall(r'\b\d+\b',next_url)[0]
          self.logger.info("Sledeca strana!!!!!!! "+url)
          yield scrapy.Request(response.urljoin(url), self.parse)
            
     


        
