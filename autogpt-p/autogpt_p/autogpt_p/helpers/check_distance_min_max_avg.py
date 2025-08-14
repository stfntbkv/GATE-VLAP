from autogpt_p.helpers.semantic_similarity import cosine_distance_str
import csv
import logging
from itertools import combinations

def get_min(distances_as_list):
    return min(distances_as_list)

def get_max(distances_as_list):
    return max(distances_as_list)

def get_avg(distances_as_list):
    return sum(distances_as_list) / len(distances_as_list)

def read_csv(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        data = [row for row in csv_reader]
    return data

def read_csv2(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        data = [row for row in csv_reader]
    return data

def calculate(row): #for similarity within one row
    result = cosine_distance_str(row[0], row[1])
    logging.info(row[0] + ' --- ' + row[1])
    logging.info(result)
    return result

def calculate2(row1, row2): # for similarity between two different rows
    print(row1, row2)
    result = cosine_distance_str(row1, row2)
    print(result)
    # logging.info(row1 + ' --- ' + row2)
    # logging.info(result)
    return result

def calculate3(row): #for similarity within a row
    result = cosine_distance_str(row[0], row[3])
    print(row[0])
    print(row[3])
    print(result)
    logging.info(row[0] + ' --- ' + row[3])
    logging.info(result)
    return result

def write_list(file_path):
    data = read_csv(file_path)
    result_list = []
    for row in data:
        result = calculate(row)
        result_list.append(result)
    return result_list

def write_list2(file_path):
    data = read_csv2(file_path)
    column_data = [row[0] for row in data]
    result_list = []
    for row1, row2 in combinations(column_data, 2):
        result = calculate2(row1, row2)
        result_list.append(result)
    return result_list

def write_list3(file_path):
    data = read_csv2(file_path)
    result_list = []
    for row in data:
        result = calculate3(row)
        result_list.append(result)
    return result_list

def write_list4(file_path):
    data_memory = read_csv2('../incremental_goal_memory/evaluation_memory.csv')
    data = read_csv2(file_path)
    column_data_memory = [row[0] for row in data_memory]
    column_data = [row[0] for row in data]
    result_list = []
    for row1 in column_data_memory:
        for row2 in column_data:
            result = calculate2(row1, row2)
            result_list.append(result)
    return result_list

def check_task_to_correction():
    logging.basicConfig(filename='similarity_task_to_correction.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("TASK TO CORRECTION")
    distance_list = write_list3('../incremental_goal_memory/evaluation_memory.csv')
    logging.info(distance_list)
    logging.info("Minimum:")
    logging.info(get_min(distance_list))
    logging.info("Maximum:")
    logging.info(get_max(distance_list))
    logging.info("Average:")
    logging.info(get_avg(distance_list))

def check_memory_to_existing_datasets():
    logging.basicConfig(filename='similarity_memory_to_existing_datasets.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # logging.info("SIMPLE GOAL TO SIMPLE GOAL")
    # distance_list = write_list4('../../data/evaluation/scenarios/evaluation_simple_goal.csv')
    # logging.info(distance_list)
    # logging.info("Minimum:")
    # logging.info(get_min(distance_list))
    # logging.info("Maximum:")
    # logging.info(get_max(distance_list))
    # logging.info("Average:")
    # logging.info(get_avg(distance_list))
    #
    # logging.info("\n")
    #
    # logging.info("SIMPLE GOAL TO SIMPLE TASK")
    # distance_list = write_list4('../../data/evaluation/scenarios/evaluation_simple_task.csv')
    # logging.info(distance_list)
    # logging.info("Minimum:")
    # logging.info(get_min(distance_list))
    # logging.info("Maximum:")
    # logging.info(get_max(distance_list))
    # logging.info("Average:")
    # logging.info(get_avg(distance_list))
    #
    # logging.info("\n")
    #
    # logging.info("SIMPLE GOAL TO COMPLEX GOAL")
    # distance_list = write_list4('../../data/evaluation/scenarios/evaluation_complex_goal.csv')
    # logging.info(distance_list)
    # logging.info("Minimum:")
    # logging.info(get_min(distance_list))
    # logging.info("Maximum:")
    # logging.info(get_max(distance_list))
    # logging.info("Average:")
    # logging.info(get_avg(distance_list))
    #
    # logging.info("\n")
    #
    # logging.info("SIMPLE GOAL TO KNOWLEDGE")
    # distance_list = write_list4('../../data/evaluation/scenarios/evaluation_knowledge.csv')
    # logging.info(distance_list)
    # logging.info("Minimum:")
    # logging.info(get_min(distance_list))
    # logging.info("Maximum:")
    # logging.info(get_max(distance_list))
    # logging.info("Average:")
    # logging.info(get_avg(distance_list))
    #
    # logging.info("\n")
    #
    # logging.info("SIMPLE GOAL TO IMPLICIT")
    # distance_list = write_list4('../../data/evaluation/scenarios/evaluation_implicit.csv')
    # logging.info(distance_list)
    # logging.info("Minimum:")
    # logging.info(get_min(distance_list))
    # logging.info("Maximum:")
    # logging.info(get_max(distance_list))
    # logging.info("Average:")
    # logging.info(get_avg(distance_list))

def check_within_dataset():
    logging.basicConfig(filename='similarity_within_dataset.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("SIMPLE GOAL")
    distance_list = write_list2('../../data/evaluation/scenarios/evaluation_simple_goal.csv')
    logging.info(distance_list)
    logging.info("Minimum:")
    logging.info(get_min(distance_list))
    logging.info("Maximum:")
    logging.info(get_max(distance_list))
    logging.info("Average:")
    logging.info(get_avg(distance_list))

    logging.info("\n")

    logging.info("SIMPLE TASK")
    distance_list = write_list2('../../data/evaluation/scenarios/evaluation_simple_task.csv')
    logging.info(distance_list)
    logging.info("Minimum:")
    logging.info(get_min(distance_list))
    logging.info("Maximum:")
    logging.info(get_max(distance_list))
    logging.info("Average:")
    logging.info(get_avg(distance_list))

    logging.info("\n")

    logging.info("COMPLEX GOAL")
    distance_list = write_list2('../../data/evaluation/scenarios/evaluation_complex_goal.csv')
    logging.info(distance_list)
    logging.info("Minimum:")
    logging.info(get_min(distance_list))
    logging.info("Maximum:")
    logging.info(get_max(distance_list))
    logging.info("Average:")
    logging.info(get_avg(distance_list))

    logging.info("\n")

    logging.info("KNOWLEDGE")
    distance_list = write_list2('../../data/evaluation/scenarios/evaluation_knowledge.csv')
    logging.info(distance_list)
    logging.info("Minimum:")
    logging.info(get_min(distance_list))
    logging.info("Maximum:")
    logging.info(get_max(distance_list))
    logging.info("Average:")
    logging.info(get_avg(distance_list))

    logging.info("\n")

    logging.info("IMPLICIT")
    distance_list = write_list2('../../data/evaluation/scenarios/evaluation_implicit.csv')
    logging.info(distance_list)
    logging.info("Minimum:")
    logging.info(get_min(distance_list))
    logging.info("Maximum:")
    logging.info(get_max(distance_list))
    logging.info("Average:")
    logging.info(get_avg(distance_list))

    logging.info("\n")

    logging.info("MEMORY")
    distance_list = write_list2('../incremental_goal_memory/evaluation_memory.csv')
    logging.info(distance_list)
    logging.info("Minimum:")
    logging.info(get_min(distance_list))
    logging.info("Maximum:")
    logging.info(get_max(distance_list))
    logging.info("Average:")
    logging.info(get_avg(distance_list))

def write_list5(file_path):
    data_memory = read_csv2('../helpers/saycan_plan_v0_l.csv')
    data = read_csv2(file_path)
    column_data_memory = [row[0] for row in data_memory]
    column_data = [row[0] for row in data]
    result_list = []
    for row1 in column_data_memory:
        for row2 in column_data:
            result = calculate2(row1, row2)
            result_list.append(result)
    return result_list

def check_task_to_correction():
    logging.basicConfig(filename='similarity_task_to_correction.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("TASK TO CORRECTION")
    distance_list = write_list3('../incremental_goal_memory/evaluation_memory.csv')
    logging.info(distance_list)
    logging.info("Minimum:")
    logging.info(get_min(distance_list))
    logging.info("Maximum:")
    logging.info(get_max(distance_list))
    logging.info("Average:")
    logging.info(get_avg(distance_list))

def check_to_saycan():
    logging.basicConfig(filename='similarity_to_saycan.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("SAYCAN TO SIMPLE GOAL")
    distance_list = write_list5('../../data/evaluation/scenarios/evaluation_simple_goal.csv')
    logging.info(distance_list)
    logging.info("Minimum:")
    logging.info(get_min(distance_list))
    logging.info("Maximum:")
    logging.info(get_max(distance_list))
    logging.info("Average:")
    logging.info(get_avg(distance_list))

    logging.info("\n")

    logging.info("SAYCAN TO SIMPLE TASK")
    distance_list = write_list5('../../data/evaluation/scenarios/evaluation_simple_task.csv')
    logging.info(distance_list)
    logging.info("Minimum:")
    logging.info(get_min(distance_list))
    logging.info("Maximum:")
    logging.info(get_max(distance_list))
    logging.info("Average:")
    logging.info(get_avg(distance_list))

    logging.info("\n")

    logging.info("SAYCAN TO COMPLEX GOAL")
    distance_list = write_list5('../../data/evaluation/scenarios/evaluation_complex_goal.csv')
    logging.info(distance_list)
    logging.info("Minimum:")
    logging.info(get_min(distance_list))
    logging.info("Maximum:")
    logging.info(get_max(distance_list))
    logging.info("Average:")
    logging.info(get_avg(distance_list))

    logging.info("\n")

    logging.info("SAYCAN TO KNOWLEDGE")
    distance_list = write_list5('../../data/evaluation/scenarios/evaluation_knowledge.csv')
    logging.info(distance_list)
    logging.info("Minimum:")
    logging.info(get_min(distance_list))
    logging.info("Maximum:")
    logging.info(get_max(distance_list))
    logging.info("Average:")
    logging.info(get_avg(distance_list))

    logging.info("SAYCAN TO MEMORY")
    distance_list = write_list5('../../data/evaluation/scenarios/evaluation_memory.csv')
    logging.info(distance_list)
    logging.info("Minimum:")
    logging.info(get_min(distance_list))
    logging.info("Maximum:")
    logging.info(get_max(distance_list))
    logging.info("Average:")
    logging.info(get_avg(distance_list))

def check_between_memory_and_evaluation():
    logging.basicConfig(filename='embedding_min_max_avg.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("OTHER ENVIRONMENT:")
    distance_list = write_list('embeddings_other_environment.csv')
    logging.info(distance_list)
    logging.info("Minimum:")
    logging.info(get_min(distance_list))
    logging.info("Maximum:")
    logging.info(get_max(distance_list))
    logging.info("Average:")
    logging.info(get_avg(distance_list))

    logging.info("OTHER OBJECTS:")
    distance_list = write_list('embeddings_other_objects.csv')
    logging.info(distance_list)
    logging.info("Minimum:")
    logging.info(get_min(distance_list))
    logging.info("Maximum:")
    logging.info(get_max(distance_list))
    logging.info("Average:")
    logging.info(get_avg(distance_list))

    logging.info("REFORMATED:")
    distance_list = write_list('embeddings_reformated.csv')
    logging.info(distance_list)
    logging.info("Minimum:")
    logging.info(get_min(distance_list))
    logging.info("Maximum:")
    logging.info(get_max(distance_list))
    logging.info("Average:")
    logging.info(get_avg(distance_list))

if __name__ == "__main__":
    # check_between_memory_and_evaluation()
    # check_within_dataset()
    # check_task_to_correction()
    # check_memory_to_existing_datasets()

    check_to_saycan()

