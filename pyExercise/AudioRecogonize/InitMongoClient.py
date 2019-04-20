import pymongo
import datetime
from pymongo import MongoClient


class InitMongoClient:

    instance = None
    DESC = None

    def __init__(self, BRIEF = "test"):
        DESC = BRIEF

    def Initialize_Client(self):
        #client = MongoClient()
        return MongoClient('localhost', 27017)
        # client = MongoClient('mongodb://localhost:27017/')
        #print("CLIENT:- ", client)

    def create_DataBase(self, Client):
        return Client['posts']

    def create_Daily_Collection(self, Dbase, date = ""):
        return Dbase['test_collection_' + date]

    def get_Current_Date(self):
        return datetime.datetime.now().strftime('%Y_%m_%d')

def create_Instance():
    instance = InitMongoClient("")
    NEW_CLIENT = instance.Initialize_Client()
    NEW_DB = instance.create_DataBase(NEW_CLIENT)
    NEW_Instance = instance.create_Daily_Collection(NEW_DB, instance.get_Current_Date())
    return NEW_Instance


#instance = create_Instance
#print(instance)

#print(table)