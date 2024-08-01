from langchain_community.embeddings import ClovaEmbeddingsV2
from env.config import EMBEDDING_CONFIG


class ClovaEmbedding:
    def __init__(self):
        self.embeddings = ClovaEmbeddingsV2(
            clova_emb_api_key=EMBEDDING_CONFIG["API_KEY"],
            clova_emb_apigw_api_key=EMBEDDING_CONFIG["GATEWAY_KEY"],
            app_id=EMBEDDING_CONFIG["ID"],
        )

    def get_embedding(self):
        return self.embeddings


if __name__ == "__main__":
    ce = ClovaEmbedding()
    print(ce.get_embedding())
