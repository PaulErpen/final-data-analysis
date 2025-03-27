import argparse
import firebase_admin
from firebase_admin import credentials, firestore
import firebase_admin.auth
from google.cloud.firestore_v1 import CollectionReference
import pandas as pd


def create_report_files(
    serviceAccountKeyPath: str,
    ratings_path: str = "./scripts/ratings.xlsx",
    pairs_path: str = "./scripts/pairs.xlsx",
) -> None:
    cred = credentials.Certificate(serviceAccountKeyPath)
    app = firebase_admin.initialize_app(cred, {"databaseURL": "-default"})

    database = firestore.client()

    ratingRef: CollectionReference = database.collection("rating")

    rating_records = []

    for doc in ratingRef.stream():
        doc_data = doc.to_dict()
        doc_data["id"] = doc.id
        rating_records.append(doc_data)

    df_ratings = pd.DataFrame.from_records(rating_records)
    df_ratings.to_excel(ratings_path)

    pairRef: CollectionReference = database.collection("pair")

    pair_records = []

    for doc in pairRef.stream():
        doc_data = doc.to_dict()
        doc_data["id"] = doc.id
        pair_records.append(doc_data)

    df_pairs = pd.DataFrame.from_records(pair_records)
    df_pairs.to_excel(pairs_path)

    firebase_admin.delete_app(app)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--serviceAccountKeyPath",
        "-p",
        type=str,
        help="The path to the service account key",
        required=True,
    )

    parsed_args = parser.parse_args()
