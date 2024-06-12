import pandas as pd
import psycopg2
import os
import sys
from dotenv import load_dotenv


class DbClass:
    def __init__(self) :
         load_dotenv()


    def connect(self):
        self.conn = None
        try:
            print('Connecting…')
            self.conn = psycopg2.connect(
                        host=os.environ['DB_PATH'],
                        database=os.environ['DB_NAME'],
                        user=os.environ['DB_USERNAME'],
                        password=os.environ['DB_PASSWORD'])
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            sys.exit(1)
        print('Povezani na bazu')
        
    def getGodine(self):
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT distinct godina FROM knjige_filtered order by godina desc;")
        except (Exception, psycopg2.DatabaseError) as error:
                print('Error: %s' % error)
                cursor.close()
                return 1
    
        rows = cursor.fetchall()
        cursor.close()
        
        return rows
    

    def getPovez(self):
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT distinct povez FROM knjige_filtered;")
        except (Exception, psycopg2.DatabaseError) as error:
                print('Error: %s' % error)
                cursor.close()
                return 1
    
        rows = cursor.fetchall()
        cursor.close()
        
        return rows
    

    def getIzdavaci(self):
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT distinct izdavac FROM knjige_filtered  order by izdavac;")
        except (Exception, psycopg2.DatabaseError) as error:
                print('Error: %s' % error)
                cursor.close()
                return 1
    
        rows = cursor.fetchall()
        cursor.close()
        
        return rows


    def getKategorije(self):
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT distinct kategorija FROM knjige_filtered order by kategorija;")
        except (Exception, psycopg2.DatabaseError) as error:
                print('Error: %s' % error)
                cursor.close()
                return 1
    
        rows = cursor.fetchall()
        cursor.close()
        
        return rows



    def getTable(self, tbl,columns):
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SELECT * FROM {tbl};")
        except (Exception, psycopg2.DatabaseError) as error:
                print('Error: %s' % error)
                cursor.close()
                return 1
    
        rows = cursor.fetchall()
        cursor.close()
        df = pd.DataFrame(rows,columns=columns)
        return df

    def close(self):
       self.conn.close()
         