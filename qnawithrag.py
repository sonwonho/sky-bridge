import json

import requests
from langchain_community.vectorstores import Milvus

from env.config import RAG_PROMPT_CONFIG
from utils.clova_completion_exe import CompletionExecutor
from utils.clova_embedding import ClovaEmbedding
from utils.open_prompt import open_prompt


class QnARag:
    def __init__(self):
        self.system_prompt = open_prompt("prompt/rag_system.txt")
        self.embeding = ClovaEmbedding()
        self.completion_executor = CompletionExecutor(is_dash=False)
        self.request_data = dict(RAG_PROMPT_CONFIG)
        self.vector_db = self._get_db()

    def _get_db(self):
        return Milvus(
            embedding_function=self.embeding.get_embedding(),
            collection_name="rag_cosine",
            index_params={
                "metric_type": "COSINE",
                "index_type": "FLAT",
                "params": {"nprobe": 10},
            },
            search_params={"metric_type": "COSINE"},
        )

    def rag(self, realquery: str):
        results = self.vector_db.similarity_search_with_score(realquery, k=5)
        reference = []

        for hit in results:
            distance = hit[-1]
            source = hit[0].metadata["source"]
            text = hit[0].page_content
            page = hit[0].metadata["page"]
            reference.append(
                {"distance": distance, "source": source, "page": page, "text": text}
            )

        preset_texts = []
        preset_texts.append({"role": "system", "content": str(self.system_prompt)})

        for ref in reference:
            preset_texts.append(
                {
                    "role": "system",
                    "content": f"reference: {ref['text']}, source: {ref['source']}, page: {ref['page']}",
                }
            )
        preset_texts.append({"role": "user", "content": realquery})
        with open("test.txt", "w") as t:
            t.write(str(preset_texts))
        request_data = self.request_data.copy()
        request_data["messages"] = preset_texts

        # LLM 생성 답변 반환
        for r in self.completion_executor.execute(request_data):
            yield r


if __name__ == "__main__":
    import time

    r = QnARag()
    start_t = time.time()
    print("".join(r.rag("정승제 강사님 어때?")))
    print(time.time() - start_t)
