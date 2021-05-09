import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics

def entropy(target_col):  # okay

    elements, counts = np.unique(target_col, return_counts=True)

    entropy = np.sum(
        [(-float(counts[i]) / float(np.sum(counts))) * np.log2(float(counts[i]) / float(np.sum(counts))) for i in
         range(len(elements))])

    return entropy


def InfoGain(data, split_attribute_name, target_name):  # okay

    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    weighted_Entropy = np.sum(
        [(float(counts[i]) / float(np.sum(counts))) * entropy(
            data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])
    information_Gain = total_entropy - weighted_Entropy
    return information_Gain


def ID3(data, features, parentdata=None, target_attribute_name=None, depth=0, max_depth=20):
    if depth >= max_depth:
        return np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

    elif len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data) == 0:  # no examples
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

            subtree = ID3(sub_data, features, data, target_attribute_name, depth=depth + 1, max_depth=max_depth)

            tree[best_feature][value] = subtree

        return (tree)


def predict(query, tree, default=1):
    for key in list(query.keys()):
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


def predict_all(data, tree):
    queries = data.iloc[:, :-1].to_dict(orient="records")
    predicted = pd.DataFrame(columns=["predicted"])
    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(query=queries[i], tree=tree)
    return predicted


def train_test_split(dataset):  # okay
    train_size = int(len(dataset) * 0.8)
    training_data = dataset.iloc[:train_size].reset_index(drop=True)
    testing_data = dataset.iloc[train_size:].reset_index(drop=True)
    return training_data, testing_data


def adaboost_train(dataset, features, nIteration, target_attribute_name):
    labels = dataset[target_attribute_name].where(dataset[target_attribute_name] == 1, -1)
    Evaluation = pd.DataFrame(labels.copy())
    Evaluation['weights'] = 1 / float(len(dataset))
    alphas = []
    models = []

    for t in range(nIteration):
        df_indecies = np.arange(len(dataset))

        weights = preprocessing.normalize([Evaluation['weights']], norm='l1')[0]
        sample_indecies = np.random.choice(df_indecies, len(df_indecies), replace=True, p=weights)

        tree = ID3(data=dataset.iloc[sample_indecies], features=features, target_attribute_name=target_attribute_name,
                   max_depth=1)
        models.append(tree)
        predictions = predict_all(data=dataset, tree=tree)
        err = 0
        Evaluation['predictions'] = predictions['predicted'].where(predictions['predicted'] == 1, -1)

        Evaluation['evaluation'] = np.where(Evaluation['predictions'] == Evaluation[target_attribute_name], 1, 0)
        Evaluation['misclassified'] = np.where(Evaluation['predictions'] != Evaluation[target_attribute_name], 1, 0)

        accuracy = float(sum(Evaluation['evaluation'])) / float(len(Evaluation['evaluation']))
        misclassification = float(sum(Evaluation['misclassified'])) / float(len(Evaluation['misclassified']))

        for ind in range(len(Evaluation['predictions'])):
            if Evaluation['predictions'][ind] != Evaluation[target_attribute_name][ind]:
                err+=weights[ind]
        for ind in range(len(Evaluation['predictions'])):
            if Evaluation['predictions'][ind] == Evaluation[target_attribute_name][ind]:
                weights[ind]*=(err/(1-err))
        # print "error: "+str(err)

        epsilon = 0.00001
        alpha = np.log((1 - err) / max(err, epsilon))
        alphas.append(alpha)

        Evaluation['weights'] *= np.exp(alpha * Evaluation['misclassified'])

        # print('The Accuracy of the {0}. model is : '.format(t + 1), accuracy * 100, '%')
        # print('The misclassification rate is: ', misclassification * 100, '%')

    return alphas, models


def adaboost_predict(test_dataset, target_attribute_name, alphas, models):
    Y_test = test_dataset[target_attribute_name].reindex(range(len(test_dataset))).where(test_dataset[classify_by] == 1, -1)

    # With each model in the self.model list, make a prediction

    accuracy = []
    predictions = []

    for i in range(len(alphas)):
        prediction = alphas[i] * predict_all(test_dataset, models[i])['predicted']
        predictions.append(prediction)
        accuracy.append(
            float(np.sum(np.sign(np.sum(np.array(predictions), axis=0)) == Y_test.values)) / float(len(predictions[0])))

    # print predictions
    fin_predictions = np.sign(np.sum(np.array(predictions), axis=0))

    # print "prediction accuracy: " + str(accuracy)
    return fin_predictions


def isNumerical(col):
    try:
        type(np.float64(col[0]))
        return True
    except ValueError:
        return False


def preproccess(dataset):
    cl1_data = dataset.loc[dataset.iloc[:, -1] == '>50K']
    cl2_data = dataset.loc[dataset.iloc[:, -1] != '>50K']
    # print len(cl1_data)
    # print len(cl2_data)
    dataset = pd.concat([cl1_data[:200], cl2_data.iloc[:200]], ignore_index=True) #.iloc[:20000]
    dataset[dataset.iloc[:, -1].name] = dataset[dataset.iloc[:, -1].name].replace({'>50K': 1, '<=50K': -1})
    dataset.fillna(method='ffill')
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    # dataset = dataset.drop('customerID', axis=1)
    datatypes = np.array(dataset.dtypes)

    for col_no in range(len(datatypes)-1):
        if isNumerical(dataset[dataset.iloc[:, col_no].name]):
            # print "col no: "+str(col_no)
            org_arr= dataset[dataset.iloc[:, col_no].name]
            temp_arr=np.array([])

            for val in org_arr:
                try:
                    val=float(val)
                except:
                    val = 0           #for empty or un recognized entries
                temp_arr = np.append(temp_arr, val)

            col_val = np.sort(np.unique(temp_arr))

            max_val = -1.0
            split_val = 0
            updated_col=pd.DataFrame()

            if len(col_val)>100:                        #if the unique vals of col is still very large then work with random 200 samples
                col_val=np.random.choice(col_val, 100)

            for val in col_val:

                df=pd.DataFrame()
                df[dataset.iloc[:, col_no].name] = temp_arr
                df[classify_by] = pd.DataFrame(dataset[dataset.iloc[:, -1].name].copy())

                df.loc[df.iloc[:, 0] > val, df.iloc[:, 0].name] = 1
                df.loc[df.iloc[:, 0] != 1, df.iloc[:, 0].name] = 0

                gain = InfoGain(data=df, split_attribute_name=dataset.iloc[:, col_no].name, target_name=classify_by)
                if gain > max_val:
                    split_val = val
                    max_val = gain
                    updated_col = df.iloc[:, 0]
            # print "split val for " + dataset.iloc[:, col_no].name + ": " + str(split_val)
            dataset[dataset.iloc[:, col_no].name]=pd.DataFrame(updated_col.copy())

    return dataset


# dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
dataset = pd.read_csv('creditcard.csv')
# dataset = pd.read_csv('adult.csv')
print  "adult DT"
classify_by = dataset.iloc[:, -1].name
dataset= preproccess(dataset=dataset)

print len(dataset)
training_data, testing_data = train_test_split(dataset)

#DT
# tree = ID3(data=training_data, features=training_data.columns[:-1], target_attribute_name=classify_by,
#                    max_depth=30)
# # predicted = np.array(predict_all(data=testing_data, tree=tree)['predicted'].astype(int))
# predicted = np.array(predict_all(data=training_data, tree=tree)['predicted'].astype(int))

#Adaboost

labels = np.array(testing_data[classify_by])
# labels = np.array(training_data[classify_by])

alphas, models = adaboost_train(training_data, training_data.columns[:-1], 3, target_attribute_name=classify_by)
#
predicted = np.array(adaboost_predict(testing_data, classify_by, alphas=alphas, models=models))
# predicted = np.array(adaboost_predict(training_data, classify_by, alphas=alphas, models=models))
conf_mat = metrics.confusion_matrix(y_true=labels, y_pred=predicted)

print conf_mat

count = 0
for i in range(len(labels)):
    if labels[i] == predicted[i]:
        count=count+1
print "Final accuracy: " + str(float(count)*100.0/float(len(labels)))+"%"