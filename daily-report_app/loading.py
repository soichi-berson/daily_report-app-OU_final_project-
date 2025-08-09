from pymongo import MongoClient
import gridfs
from joblib import load
from tensorflow.keras.models import load_model
import pandas as pd
from datetime import datetime

import os

class ModelLoader:
    """
    Handles loading of machine learning models and scalers stored in MongoDB GridFS.

    Connects to the specified MongoDB database and provides methods
    to restore files from GridFS to the local filesystem, then load them
    so they are ready to use directly in Python for clustering (KMeans) and time series (LSTM) tasks.

    Attributes
    ----------
    kmeans_scaler : object
        The fitted scaler for KMeans features (loaded after calling load_kmeans_scaler()).
    kmeans_model : object
        The fitted KMeans clustering model.
    lstm_scaler : object
        The fitted scaler for LSTM model inputs.
    lstm_model : object
        The loaded LSTM model.
    """

    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="model_storage"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.fs = gridfs.GridFS(self.db)
        print(f"✅ Connected to MongoDB GridFS: {db_name}")

        self.kmeans_scaler = None
        self.kmeans_model = None
        self.lstm_scaler = None
        self.lstm_model = None

    def _restore_file(self, filename, local_path):
        file_data = self.fs.find_one({'filename': filename})
        if file_data:
            with open(local_path, 'wb') as f:
                f.write(file_data.read())
            print(f"✅ Restored '{filename}' to '{local_path}'")
        else:
            raise FileNotFoundError(f"❌ File '{filename}' not found in GridFS.")

    def load_kmeans_scaler(self, gridfs_name='scaler_final.pkl', local_path='scaler_final_restored.pkl'):
        self._restore_file(gridfs_name, local_path)
        self.kmeans_scaler = load(local_path)

    def load_kmeans_model(self, gridfs_name='kmeans_model_k4_final.pkl', local_path='kmeans_model_restored.pkl'):
        self._restore_file(gridfs_name, local_path)
        self.kmeans_model = load(local_path)

    def load_lstm_scaler(self, gridfs_name='audience_scaler_LSTM.save', local_path='lstm_scaler_restored.save'):
        self._restore_file(gridfs_name, local_path)
        self.lstm_scaler = load(local_path)

    def load_lstm_model(self, gridfs_name='final_lstm_model.h5', local_path='final_lstm_model_restored.h5'):
        self._restore_file(gridfs_name, local_path)
        self.lstm_model = load_model(local_path)

    def load_all(self):
        """Convenience method to load everything at once."""
        self.load_kmeans_scaler()
        self.load_kmeans_model()
        self.load_lstm_scaler()
        self.load_lstm_model()
        print("✅ All models and scalers successfully loaded.")





class DataFetcher:
    """
    Retrieve PVF and IBT data from MongoDB collections as pandas DataFrames.

    This class connects to two MongoDB databases:
    - "pvf_database" for PVF data used in KMeans clustering.
    - "ibt_database" for IBT audience data used in LSTM forecasting.

    Attributes
    ----------
    client : pymongo.MongoClient
        Connection to the MongoDB instance.
    pvf_db : pymongo.database.Database
        Reference to the PVF database.
    ibt_db : pymongo.database.Database
        Reference to the IBT database.
    """


    def __init__(self, mongo_uri="mongodb://localhost:27017/"):
        self.client = MongoClient(mongo_uri)
        self.pvf_db = self.client["pvf_database"]
        self.ibt_db = self.client["ibt_database"]

    def _parse_date(self, yyyymmdd):
        year = yyyymmdd // 10000
        month = (yyyymmdd % 10000) // 100
        day = yyyymmdd % 100
        return datetime(year, month, day)

    def get_dataframes(self, yyyymmdd, base_date=datetime(2025, 5, 14)):
        target_date = self._parse_date(yyyymmdd)

        # --- PVF: Today
        pvf_collection = self.pvf_db["data"]
        today_data = list(pvf_collection.find({'Date of Activity': target_date}, {'_id': 0}))
        today_kmeans_df = pd.DataFrame(today_data)

        # --- PVF: Base
        base_data = list(pvf_collection.find({'Date of Activity': base_date}, {'_id': 0}))
        base_df = pd.DataFrame(base_data)

        # --- IBT: LSTM
        ibt_collection = self.ibt_db["audience_data"]
        lstm_cursor = ibt_collection.find({"DATE": target_date}, {'_id': 0})
        lstm_df = pd.DataFrame(lstm_cursor)

        return today_kmeans_df, base_df, lstm_df
