from __future__ import annotations

from autogpt_p.helpers.semantic_similarity import cosine_distance, get_embeddings, cosine_distance_str
from pddl.problem import *
import pandas as pd

class IncrementalGoalMemory:

    def __init__(self, filename='known_pairs.csv'):
        """
        Initialize the Memory with the pair of users request and users modified goal as a string
        """

        self.filename = filename
        self.reset_csv()
        self.load_memory()

    def load_memory(self):
        self.memory = pd.read_csv(self.filename)
    def save_memory(self, filename=None):
        if filename is not None:
            self.memory.to_csv(filename, index=False)
        else:
            self.memory.to_csv(self.filename, index=False)

    def add_pair(self, new_request, new_user_modified_goal):
        '''
        Add a new pair of user request and user modified goal
        :param new_request: users request in natural language
        :param new_user_modified_goal: from correction tool detected modified goal
        :return:
        '''
        new_user_modified_goal = str(new_user_modified_goal)
        new_embedded_request = [get_embeddings(new_request)]

        new_row = {'request': new_request, 'user_modified_goal_str': new_user_modified_goal, 'embedded_request': new_embedded_request}
        new_row2 = pd.DataFrame(new_row)
        self.memory = pd.concat([self.memory, new_row2])

    @property
    def get_pairs(self):
        """
        Retrieve all stored pairs from the CSV file using Pandas.
        """
        return self.memory

    def reset_csv(self):
        '''
        Reset the csv-file
        '''
        self.memory = pd.DataFrame(columns=['request', 'user_modified_goal_str', 'embedded_request'])

    def get_k_closest_requests(self, k: int, new_request: str):
        self.save_copy_to('distances.csv')
        distance_csv = pd.read_csv('distances.csv')
        distance_csv['cosine_distance_to_new_request'] = None  # Initialisiere die Spalte mit None
        distance_csv.to_csv('distances.csv', index=False)

        new_request_embedded = get_embeddings(new_request)
        # berechne cosinus distanz der neuen Anfrage zu jedem Eintrag und schreib es in distances_csv
        distance_csv['cosine_distance_to_new_request'] = distance_csv['embedded_request'].apply(cosine_distance, args=(new_request_embedded,))
        # print('nach funktion')
        # print(distance_csv['cosine_distance_to_new_request'])
        #sortiere nach cosinusdistanz
        sorted_distances_csv = distance_csv.sort_values(by='cosine_distance_to_new_request', ascending=False)
        #gib die top k zurück
        top_k_data = sorted_distances_csv.head(k)
        return top_k_data

    def k_closest_requests_as_str(self, k: int, new_request: str):
        x = self.get_k_closest_requests(k, new_request)
        y = self.memory
        number_of_rows = y.shape[0]
        # print("*"*100)
        # print(number_of_rows)
        k_pairs = []
        for i in range(min(k, number_of_rows)):
            request = x.iloc[i].iloc[0]
            # print(request)
            user_modified_goal = x.iloc[i].iloc[1]
            closest_similarity = x.iloc[0].iloc[3]
            similarity = x.iloc[i].iloc[3] # Schwellwert für Mindestmaß an Ähnlichkeit
            if closest_similarity > 0.95: #new threshold question
                k_pairs.append("Q" + str(i + 1) + ": " + request + "\n" + "A: " + '(:goal ' + user_modified_goal + ")")
                break
            else:
                if similarity > 0.7:
                    # k_pairs.append("Example " + str(i+1) +": " +request + " -> " + '(:goal '+ user_modified_goal +")")
                    k_pairs.append("Q" + str(i + 1) + ": " + request + "\n" + "A: " + '(:goal ' + user_modified_goal + ")")

        k_closest_requests_as_str = "\n".join(k_pairs)
        # print('k-closest_requests_as_str wurde aufgerufen! Memory wird also verwendet')
        k_closest_requests_as_str = 'These are examples from previous user interactions:' + '\n' + k_closest_requests_as_str + '\n' + 'Consider these example in your answer'
        return k_closest_requests_as_str

    def save_copy_to(self, new_filename):
        self.memory.to_csv(new_filename, index=False)

if __name__ == "__main__":
    # Beispiele printen und testen
    memory = IncrementalGoalMemory()
    # memory.reset_csv()
    # memory.save_memory()
    # memory.add_pair("request3", (And([Predicate("inhand", [Object("banana", Type("banana", [])),
    #                                                               Object("robot", Type("robot", []))])])))
    #
    # memory.add_pair("Bring an apple", (And([Predicate("inhand", [Object("apple", Type("apple", [])),
    #                                                              Object("robot", Type("robot", []))])])))
    #
    # memory.add_pair("Cut the apple", (And([Predicate("inhand", [Object("apple", Type("apple", [])),
    #                                                             Object("robot", Type("robot", []))])])))
    #
    # memory.add_pair("Bring a glass of milk", (And([Predicate("inhand", [Object("milk", Type("milk", [])),
    #                                                                     Object("robot", Type("robot", []))])])))
    # memory.add_pair("Bring a banana", And([Predicate("inhand", [Object("milk", Type("milk", [])),
    #                                                                     Object("robot", Type("robot", []))])]))
    # memory.save_memory()

    # x = memory.k_closest_requests_as_str(3, 'Bring a banana')
    x = memory.k_closest_requests_as_str(3, "Help me prepare a salad with tomatoes")
    print("=" * 100)
    # x = memory.memory
    print(x)
    # print(type(x.iloc[3].iloc[2]))
    #
    # print(x.iloc[3].iloc[2])

    # y = memory.is_request_already_existing('I want to watch TV')
    # z = memory.get_k_closest_requests(1, 'I want to watch TV').iloc[0].iloc[1]
    #
    # print(y)
    # print("-"*100)
    # print(z)
    # print(type(z))
