import json
import pandas as pd

def read_large_jl_file_and_convert_to_json(jl_file_path, json_file_path):
    data_list = []
    conditions = ["control", "depression", "adhd", "anxiety", "autism", "bipolar", "eating", "ocd", "ptsd", "schizophrenia"]
    condition_counters = {condition: 0 for condition in conditions}

    with open(jl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_data = json.loads(line)
                # Determine the condition label
                condition_label = None
                for condition in conditions:
                    if condition in json_data['label']:
                        condition_label = condition
                        break
                if condition_label is None: 
                    continue
                # Check if the limit for this condition is not exceeded
                if condition_counters[condition_label] < 100:
                    condition_counters[condition_label] += 1
                    json_data['condition'] = condition_label
                    data_list.append(json_data)

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in line: {line}")
                print(e)

            # Stop if 100 entries for each condition are collected
            if all(count >= 100 for count in condition_counters.values()):
                break

    # Convert the list of JSON data to a DataFrame
    df = pd.DataFrame(data_list)

    # Write the DataFrame to a JSON file
    df.to_json(json_file_path, orient='records', lines=True)
    print(f'Conversion complete. JSON file saved at {json_file_path}')

if __name__ == "__main__":
    jl_file_path = 'SMHD/SMHD_train.jl'
    json_file_path = 'SMHD/train_smhd_limited.json'

    read_large_jl_file_and_convert_to_json(jl_file_path, json_file_path)
