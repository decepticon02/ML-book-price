# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import psycopg2

class KnjigeScraperPipeline:

    def __init__(self):

        hostname = 'localhost'
        username = 'postgres'
        password = 'postgres'
        database = 'knjigedb'

        self.connection = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
        
        
        self.cur = self.connection.cursor()
    
        if(input("Brisanje tabele? (y/n):")=='y'):
            self.cur.execute("""DROP TABLE IF EXISTS knjige""")
        
        self.cur.execute("""
                CREATE TABLE IF NOT EXISTS knjige(
                    id SERIAL,
                    sifra VARCHAR UNIQUE,
                    title text,
                    autor text,
                    kategorija VARCHAR(255),
                    izdavac text,
                    povez text,
                    godina VARCHAR(10),            
                    format text,
                    strana int,
                    opis text,
                    cena real
                )
                """)
        


    def process_item(self, item, spider):
        try:
            self.cur.execute("""INSERT INTO knjige 
                            (sifra, title, autor,kategorija,
                            izdavac,povez,godina,format,
                            strana,opis,cena) 
                            values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (sifra) DO UPDATE SET
                            title = EXCLUDED.title,
                            autor = EXCLUDED.autor,
                            kategorija = EXCLUDED.kategorija,
                            izdavac = EXCLUDED.izdavac,
                            povez = EXCLUDED.povez,
                            godina = EXCLUDED.godina,
                            format = EXCLUDED.format,
                            strana = EXCLUDED.strana,
                            opis = EXCLUDED.opis,
                            cena = EXCLUDED.cena""", (
                                item["sifra"],
                                item["title"],
                                item["autor"],
                                item["kategorija"],
                                item["izdavac"],
                                item["povez"],
                                item["godina"],
                                item["format"],
                                item["strana"],
                                item["opis"],
                                item["cena"],
                            ))

            self.connection.commit()

        except psycopg2.Error as e:
            print("Error occurred:", e)
            self.connection.rollback()

        return item

    def close_spider(self, spider):        
        self.cur.close()
        self.connection.close()
