import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--file', type=str, default='data/classes/random_classes.json', help='Path to objects file')

    args = parser.parse_args()
    file = args.file

    with open(file) as f:
        data_dict = json.load(f)
    with open(file, 'w') as f:
        json.dump(data_dict, f, indent=4)

