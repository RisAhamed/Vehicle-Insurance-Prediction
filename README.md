# Mlops-with-Mongo-DB

python -m pip install "pymongo[srv]"==3.6


from pymongo.mongo_client import MongoClient

uri = "mongodb+srv://riswanahamed38:<db_password>@cluster0.8fvbl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

mongodb+srv://riswanahamed38:<db_password>@cluster0.8fvbl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0

artifacts/
└── timestamp_folder/
    └── data_ingestion/
        ├── unzipped_data/
        │   └── visa_data.csv
        ├── raw_data/
        │   └── visa_data.csv
        └── ingested_data/
            ├── train/
            │   └── train.csv
            └── test/
                └── test.csv