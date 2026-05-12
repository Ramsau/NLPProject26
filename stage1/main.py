import csv


def count_occurrences(list):
    list_count = {}
    for item in list:
        if item in list_count:
            list_count[item] += 1
        else:
            list_count[item] = 1
    return sorted(list_count.items(), key=lambda x: x[1], reverse=True)

def main():
    with open('../dataset.csv', 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)

    # discard header
    data.pop(0)
    train_idx = int(len(data) * 0.8)
    train_data = data[:train_idx]
    test_data = data[train_idx:]

    train_facts = [entry[0] for entry in train_data if entry[1] == 'Fact']
    train_opinions = [entry[0] for entry in train_data if entry[1] == 'Opinion']

    train_facts_count = count_occurrences(train_facts)
    train_opinions_count = count_occurrences(train_opinions)

    print(train_facts_count)
    print(train_opinions_count)





if __name__ == '__main__':
    main()
