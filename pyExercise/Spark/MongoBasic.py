import pymongo
import datetime
from pymongo import MongoClient
client = MongoClient()

client = MongoClient('localhost', 27017)
# client = MongoClient('mongodb://localhost:27017/')
#print("CLIENT:- ", client)

#db = client.test_database
db = client['posts']
#print("DATABASE:- ", db)
collection = db.test_collection
#collection = db.test_collection
#collection = db['test_collection']
#print("COLLECTION:- ", collection)




post = {"author": "Mike",
         "text": "My first blog post!",
         "tags": ["mongodb", "python", "pymongo"],
         "date": datetime.datetime.utcnow()}

print(post)

posts = db.posts
#post_id = collection.insert_one(post).inserted_id

print(posts.find_one())