import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMClassifier

class TopicPredictor:
    def __init__(self):
        # Load model
        self.model = pickle.load(open("model_lgb.pkl", "rb"))

        # Load content and topics
        self.df_content = pd.read_csv("embedding_content_dataset.csv")
        self.df_topics = pd.read_csv("embedding_topics_dataset.csv")

        # Rename embeddings cols (0–127) to 0_content ... 127_content
        content_embedding_cols = [f"{i}_content" for i in range(128)]
        self.df_content.rename(columns={old: new for old, new in zip(self.df_content.columns[:128], content_embedding_cols)}, inplace=True)
        
        # Rename specific columns
        self.df_content.rename(columns={
            "language": "language_content",
            "id": "id_content"
        }, inplace=True)
        
        # Rename topics columns from (0–127) to 0_topic ... 127_topic
        topic_embedding_cols = [f"{i}_topic" for i in range(128)]
        self.df_topics.rename(columns={old: new for old, new in zip(self.df_topics.columns[:128], topic_embedding_cols)}, inplace=True)
        
        # Rename specific columns
        self.df_topics.rename(columns={
            "language": "language_topic",
            "id": "id_topic"
        }, inplace=True)

    def predict(self, content_id, threshold=0.5):
        """
        Given a content_id, return a list of relevant topic_ids.
        """

        # Get row content
        content_row = self.df_content[self.df_content["id_content"] == content_id]
        if content_row.empty:
            return []

        # Expand content against topics
        content_features = content_row.iloc[0]  # convert to Series
        #content_features = content_row.drop("id")

        repeated_content = pd.DataFrame([content_features.values] * len(self.df_topics), columns=content_features.index)

        # Combine the datasets
        all_candidates = pd.concat([repeated_content.reset_index(drop=True), self.df_topics.reset_index(drop=True)], axis=1)

        categorical_cols = [
            "kind",
            "language_content",
            "copyright_holder",
            "license",
            "category",
            "language_topic"
        ]
        
        for col in categorical_cols:
            all_candidates[col] = all_candidates[col].astype("category")

        # Drop unused columns
        X = all_candidates.drop(columns=["id_content", "id_topic"])

        # Make the prediction
        preds = self.model.predict_proba(X)[:, 1]  # Probability of class 1

        # Get the topic_ids where the prediction is above the threshold
        selected_indices = np.where(preds > threshold)[0]

        # Get the corresponding topic_ids
        selected_topic_ids = self.df_topics.iloc[selected_indices]["id_topic"].tolist()

        return selected_topic_ids