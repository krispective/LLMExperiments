import pinecone
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import json

index_name = "genaihackathon"
pinecone_api_key = "<api_key>"
# Initialize a Pinecone client with your API key
pc = Pinecone(api_key=pinecone_api_key)

# Uncomment the following when you begin to use pinecone
# index = pc.Index(index_name)

json_file_path = "app/context.json"

with open(json_file_path, 'r') as j:
     dict_doc = json.loads(j.read())



def fetch_best_context(query):
    index = pc.Index(index_name)
    # Convert the query into a numerical vector that Pinecone can search with
    query_embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query],
        parameters={
            "input_type": "query"
        }
    )

    # Search the index for the three most similar vectors
    results = index.query(
        namespace=index_name,
        vector=query_embedding[0].values,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    print('Results obtained from Pinecone vector database.')

    top_result_uuid = results['matches'][0]['id']

    print(dict_doc[top_result_uuid])
    return dict_doc[top_result_uuid]

