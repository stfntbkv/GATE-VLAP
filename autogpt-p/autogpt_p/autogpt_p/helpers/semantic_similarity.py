from typing import List

from sklearn.metrics.pairwise import cosine_similarity
import ast
from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("BAAI/bge-m3")

def get_embeddings(request: str):
    response_as_array = model.encode(request)
    response = response_as_array.tolist()
    return response

def cosine_distance(embedded_request1: str, embedded_request2: List[float]):
    embedded_request1 = ast.literal_eval(embedded_request1)
    # print("="*100)
    # print(embedded_request1, "\n", embedded_request2)
    # print(type(embedded_request1), "\n", type(embedded_request2))

    request1 = [embedded_request1]
    request2 = [embedded_request2]
    return cosine_similarity(request1, request2)[0][0]
def cosine_distance_str(request1: str, request2: str):
    # print("="*100)
    # print(request1, "\n", request2)
    request1 = [get_embeddings(request1)]
    request2 = [get_embeddings(request2)]
    return cosine_similarity(request1, request2)[0][0]

if __name__ == "__main__":
    # print(get_embeddings('Bring an apple'))
    # print("a",cosine_distance_str('and (on fork0 table0) (inhand apple0 human0)', 'and (inhand apple0 human0) (on fork0 table0)'))
    # print("b",cosine_distance_str('I would like to eat walnuts', 'I would like to eat peanuts'))
    # print("c",cosine_distance_str('I want to eat walnuts', 'Bring me the peanuts'))
    # print("d", cosine_distance_str('I would like to eat walnuts', 'I want to snack some walnuts'))
    # print("e", cosine_distance_str('I am freezing', 'I feel dizzy i need water and fresh air'))
    # x = get_embeddings("Bring an apple")
    print("f", cosine_distance_str('The air is stuffy', 'Open the window and the door'))
    print("g", cosine_distance_str('The air is stuffy', 'The plant looks dry'))
    # y = get_embeddings("Bring an apple")
    # print(type(x), type(y))
    # cosine_distance(x,y)
    # print(cosine_distance(x,y))
    # pass
