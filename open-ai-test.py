import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential

endpoint = "https://prasa-mbkq18is-eastus2.cognitiveservices.azure.com/"
model_name = "text-embedding-3-small"
deployment = "embedding-model"

api_version = "2024-02-01"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=""
)

# Test the connection
try:
    # Test with embedding model
    response = client.embeddings.create(
        input="test query",
        model="embedding-model"
    )
    print("Connection successful!")
   
    
except Exception as e:
    print(f"Error: {e}")
    
response = client.embeddings.create(
    input=["first phrase","second phrase","third phrase"],
    model=deployment
)

for item in response.data:
    length = len(item.embedding)
    print(
        f"data[{item.index}]: length={length}, "
        f"[{item.embedding[0]}, {item.embedding[1]}, "
        f"..., {item.embedding[length-2]}, {item.embedding[length-1]}]"
    )
print(response.usage)

# Test completion model
def test_completion():
    response = client.chat.completions.create(
        model="gpt-4o",  # Your deployment name
        messages=[
            {"role": "system", "content": "You are a SQL query generator."},
            {"role": "user", "content": "Generate a SELECT query for customer data"}
        ],
        max_tokens=150,
        temperature=1.0,
        top_p=1.0
    )
    print("Completion test successful")
    return response.choices[0].message.content

sql_query = test_completion()
print(sql_query)