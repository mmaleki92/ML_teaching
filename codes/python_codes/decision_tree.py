from math import log2

def calculate_entropy(data, target_attribute):
    """ Calculate the entropy of a dataset for the target attribute. """
    value_counts = {}
    for row in data:
        label = row[target_attribute]
        if label not in value_counts:
            value_counts[label] = 0
        value_counts[label] += 1

    entropy = 0.0
    for key in value_counts:
        probability = value_counts[key] / len(data)
        entropy -= probability * log2(probability)
    return entropy

def calculate_information_gain(data, split_attribute, target_attribute):
    """ Calculate the information gain of a split attribute. """
    total_entropy = calculate_entropy(data, target_attribute)
    attribute_values = set(row[split_attribute] for row in data)

    weighted_entropy = 0
    for value in attribute_values:
        subset = [row for row in data if row[split_attribute] == value]
        probability = len(subset) / len(data)
        subset_entropy = calculate_entropy(subset, target_attribute)
        weighted_entropy += probability * subset_entropy

    information_gain = total_entropy - weighted_entropy
    return information_gain

def build_decision_tree(data, attributes, target_attribute):
    """ Build a decision tree using recursion. """
    # If all target labels are the same, return that label
    if len(set(row[target_attribute] for row in data)) == 1:
        return data[0][target_attribute]

    # If no more attributes to split, return the most common target label
    if not attributes:
        return max(set(row[target_attribute] for row in data), key=lambda label: [row[target_attribute] for row in data].count(label))

    # Select the attribute with the highest information gain
    best_attribute = max(attributes, key=lambda attr: calculate_information_gain(data, attr, target_attribute))
    tree = {best_attribute: {}}

    # Remove the best attribute from the list of attributes
    attributes = [attr for attr in attributes if attr != best_attribute]

    # Build a branch for each value of the best attribute
    for value in set(row[best_attribute] for row in data):
        subset = [row for row in data if row[best_attribute] == value]
        subtree = build_decision_tree(subset, attributes, target_attribute)
        tree[best_attribute][value] = subtree

    return tree

# Sample dataset
dataset = [
    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'No'},
    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong', 'PlayTennis': 'No'},
    {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
]

attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
target_attribute = 'PlayTennis'

# Building the decision tree
decision_tree = build_decision_tree(dataset, attributes, target_attribute)
print(decision_tree)
