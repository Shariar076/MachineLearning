import pandas as pd
import numpy as np
from pprint import pprint


dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
# dataset = pd.read_table('weather', sep='\t',header=None, names=['OUTLOOK', 'TEMPERATURE', 'HUMIDITY', 'WINDY', 'PLAY'])
# dataset = pd.read_csv('zoo.csv')
# dataset = dataset.drop('animal_name', axis=1)
dataset = dataset.sample(frac=1).reset_index(drop=True)
dataset = dataset.drop('tenure', axis=1)
dataset = dataset.drop('MonthlyCharges', axis=1)
dataset = dataset.drop('TotalCharges', axis=1)
dataset = dataset.drop('customerID', axis=1)

classify_by=dataset.iloc[:,-1].name

def entropy(target_col): #okay

    elements, counts = np.unique(target_col, return_counts=True)

    entropy = np.sum(
        [(-float(counts[i]) / float(np.sum(counts))) * np.log2(float(counts[i]) / float(np.sum(counts))) for i in range(len(elements))])

    return entropy


def InfoGain(data, split_attribute_name, target_name=classify_by): #okay

    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_Entropy = np.sum(
        [(float(counts[i]) / float(np.sum(counts))) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])
    information_Gain = total_entropy - weighted_Entropy
    return information_Gain


def ID3(data, features,parentdata=None, target_attribute_name=classify_by, depth=0, max_depth=20):
    if depth >= max_depth :
        return np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

    elif len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data) == 0:  #no examples
        return np.unique(parentdata[target_attribute_name])[
            np.argmax(np.unique(parentdata[target_attribute_name], return_counts=True)[1])]

    elif len(features) == 0:
        return np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
    else:
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature: {}}

        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):

            sub_data = data.where(data[best_feature] == value).dropna()

            subtree = ID3(sub_data, features, data, target_attribute_name, depth=depth+1, max_depth=max_depth)

            tree[best_feature][value] = subtree

        return (tree)


def predict(query, tree, default=1):

    for key in list(query.keys()):
        print key
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default
            result = tree[key][query[key]]

            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


def test(data, tree):

    queries = data.iloc[:, :-1].to_dict(orient = "records")
    predicted = pd.DataFrame(columns=["predicted"])

    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(query=queries[i], tree=tree)
    print('The prediction accuracy is: ', (float(np.sum(predicted["predicted"] == data[classify_by]))/len(data))*100, '%')


def train_test_split(dataset): #okay
    train_size=int(len(dataset)*0.8)
    training_data = dataset.iloc[:train_size].reset_index(drop=True)
    testing_data = dataset.iloc[train_size:].reset_index(drop=True)
    return training_data, testing_data


training_data, testing_data = train_test_split(dataset)


tree = ID3(data=training_data, features=training_data.columns[:-1], target_attribute_name=classify_by, max_depth=2)

pprint(tree)

test(testing_data, tree)
