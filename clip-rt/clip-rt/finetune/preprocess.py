import os
import json
import csv
import argparse

parser = argparse.ArgumentParser(description="clip-rt finetuning preprocessing")
parser.add_argument("--data-path", type=str, help="the path for data e.g., /home/work/clip_rt_in_domain_data")
args = parser.parse_args()


# load json data
json_path = os.path.join(args.data_path, 'clip_rt_in_domain_data.json')
all_samples = json.load(open(json_path, 'r'))

# save csv path
csv_path = os.path.join(args.data_path, 'clip_rt_in_domain_data.csv')

# json to map action class and natural language supervision
action_to_label = json.load(open("../docs/action_to_label_finetune.json", "r"))
label_to_language = json.load(open("../docs/label_to_language_finetune.json", "r"))


with open(csv_path, 'w', newline='') as f:
    csv_out = csv.writer(f, delimiter=',')
    csv_out.writerow(['filepath','caption','supervision','label'])
    image_path_prefix = os.path.join(args.data_path, 'images')

    for idx, sample in enumerate(all_samples):
        if idx % 1000 == 0:
            print(idx)

        try:
            action = str(sample['action'])
            label = action_to_label[action]
            command = label_to_language[label]

            item = []
            item.append(os.path.join(image_path_prefix, sample['image_id']))

            # caption
            caption = "what motion should the robot arm perform to complete the instruction '{}'?".format(sample['instruction'])
            item.append(caption)

            item.append(command)

            # label
            item.append(label)
            csv_out.writerow(item)

        except KeyError:
            print('skip abnormal action')
            continue