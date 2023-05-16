import redis
from redis.commands.search.query import Query
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import time
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from gpt.general import get_embedding
from typing import Mapping
import os

redis_host = os.getenv("REDIS_HOST")
redis_port = os.getenv("REDIS_PORT")
client = redis.Redis(host=redis_host, port=redis_port, db=0)

schema = (
    TextField("content"),
    VectorField(
        "vec",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": 1536,
            "DISTANCE_METRIC": "COSINE",
        },
    ),
)

index_name = str(int(time.time() * 1000_000))
print("INDEX: ", index_name)

# Create Index
client.ft(index_name).create_index(
    fields=schema,
    definition=IndexDefinition(prefix=[f"{index_name}:"], index_type=IndexType.HASH),
)
pipeline = client.pipeline(transaction=False)

sentences = [
    "今日はいい天気ですね",
    "あの人はとても優しいです",
    "うさぎはとても可愛いです",
    "あのワインはとても美味しいです",
    "彼の叔父はとても有名な人です",
]

# Add documents to redis
i = 0
for s in sentences:
    embedding = get_embedding(s)
    pipeline.hset(f"{index_name}:{i}", mapping={
        "content": s,
        "vec": np.array(embedding, dtype=np.float32).tobytes()
    })
    i += 1
pipeline.execute()

embedding = get_embedding("あの人のお父さんはとても有名な人です")
params_dict: Mapping[str, str] = {
    "vec": np.array(embedding).astype(dtype=np.float32).tobytes()
}

# Query
k = 3
base_query = f"*=>[KNN {k} @vec $vec AS vector_score]"
return_fields = ["content", "vector_score"]
redis_query = Query(base_query).return_fields(*return_fields).sort_by("vector_score").paging(0, k).dialect(2)
results = client.ft(index_name).search(redis_query, params_dict)

for d in results.docs:
    print(d.id)
    print(d.content, d.vector_score)