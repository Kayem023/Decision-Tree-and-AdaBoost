import csv
import math
import random
import numpy as np
import pandas as pd
import sys, getopt


# Entropy , Gain 



'''
Entropy of a binary classifier , 
Here q = probability of positive/negative sample
Outputs entropy = -qlog2q-(1-q)log2(1-q)
'''
def B(q):
    if q in [0,1]:
        return 0
    entropy = q*math.log2(q)+(1-q)*math.log2(1-q)
    return -entropy


'''
Calculates gain from positive and negative sample frequency
p = positive sample frequency
n = negative sample frequency
'''


def Gain(p,n):
    tp = 0
    tn = 0
    gain = 0
    for pn in zip(p,n):
        pk = pn[0]
        nk = pn[1]
        if pk != 0 and nk != 0:
            gain = gain + (pk+nk)* B(pk/(pk + nk))
        tp = tp + pk
        tn = tn + nk
    gain = -gain/(tp+tn) 
    gain = gain + B(tp/(tp+tn))
    return gain
    
    
# Tree Structure:

'''
A traditional tree structure with parent child relation
parent = parent of a tree-node
attribute = the attribute on which branch are made to child nodes
condition = condition of attribute for branching from parent node
answer = If a tree-node is leaf node then answer = 0 means positive , answer = 1 means negative
'''
class Tree:
    def __init__(self):
        self.parent = None
        self.attribute = 0
        self.condition = None
        self.answer = None
        self.children = {}
        
# Global Variables


'''File list'''

Files = [
    '/home/kayem/ML Asignment 1 Decision Tree/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv',
    '/home/kayem/ML Asignment 1 Decision Tree/creditcardfraud/creditcard.csv',
    '/home/kayem/ML Asignment 1 Decision Tree/Dataset 2/adult.data',
    '/home/kayem/ML Asignment 1 Decision Tree/Dataset 2/adult.test'
]

'''Attribute List'''

Attribute = [
    ['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges','Churn'],
    ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"],
    ['age','workclass','fnlwgt','education','education num','marital status','occupation','relationship','race','sex','capital gain','capital loss','hours per week','native-country','Salary Range'],
    ['age','workclass','fnlwgt','education','education num','marital status','occupation','relationship','race','sex','capital gain','capital loss','hours per week','native-country','Salary Range']
]

'''Types of attribute'''

Att_Type = [
    ['C','C','C','C','R','C','C','C','C','C','C','C','C','C','C','C','C','R','R','C'],
    ['R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','C'],
    ['R','C','R','C','R','C','C','C','C','C','R','R','R','C','C'],
    ['R','C','R','C','R','C','C','C','C','C','R','R','R','C','C']
]

'''Non-categorical attribute list'''

Non_Categorical = [
    [4,17,18],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    [0,2,4,10,11,12],
    [0,2,4,10,11,12]
]



Dataset_ID = [1,2,3,4]      # dataset-id = 1- Telco customer churn, 2- credit-card fraud, 3- adult-train, 4-adult-test


All_Dataset = [1,2,3,4]     # ids of all datasets 
First_lineRemove = [1,2,4]  # datasets ids for which first line is not needed
First_ColumnRemove = [1]    # datasets ids for which first column is not needed
Separate_Data = [4]         # datasets ids for which training and testing files are separated like adult dataset
Unbalanced_Data = [2]       # datasets on which positive and negative are unbalanced
last_CharRemove = [4]       # datasets ids for which last character is a delimiter which should not be included
last_char = {4:'.'}         # mapping of last charater which should be removed
map_binary = {1:{'Yes':1,'No':0}, 2: {'"0"':0, '"1"':1}, 3:{ '<=50K':0, '>50K':1}, 4:{ '<=50K':0, '>50K':1}} # mapping of prdiction level







# Data Preprocessing

'''
Return the data-samples, and attribute-details
Here data = File-name of the data
dataset-id = 1- Telco customer churn, 2- credit-card fraud, 3- adult-train, 4-adult-test
attribute = list of attributes
att_type = Types of attribute
attribute_detail = attribute-details (attribute : set of values)
takeMiss = True - consider the missing samples / False - ignore the missing samples
equiprobable = False - Taking all samples / True - Take a balanced small subset of samples
maxN = maximum number of positive/negative samples if equiprobable = True
'''


def getData(data, dataset_id, attribute, att_type, attribute_detail, takeMiss,equiprobable=False,maxN=0):
    if dataset_id not in Unbalanced_Data:
        equiprobable = False
    if dataset_id not in  Separate_Data:
        attribute_detail = {}
        for field in attribute:
            attribute_detail[field] = {}
    else:
        equiprobable = False
    if equiprobable:
        takeMiss = False
    attribute_detail[attribute[-1]] = map_binary[dataset_id]
    Has_missing = {}
    idx = 0
    for field in attribute:
        Has_missing[field] = []
    examples = []
    positive_examples = []
    negative_examples = []
    pos_count = 0
    neg_count = 0
    with open(data, 'r') as f:
        if dataset_id in First_lineRemove:
            f.readline()
        if dataset_id in All_Dataset:
            idx = 0
            for line in f.readlines():
                if dataset_id in last_CharRemove:
                    line = line.split(last_char[dataset_id])[0]
                line = line.strip()
                if len(line) > 0 :
                    example = []
                    idx_v = 0
                    if dataset_id in First_ColumnRemove:
                        idx_v = 1
                    values = line.split(',')
                    flag = True
                    for i in range(len(values)-idx_v):
                        value = values[idx_v + i].strip()
                        if len(value) == 0 or value in ['?']:
                            Has_missing[attribute[i]].append(idx)
                            value = 0
                            if not takeMiss:
                                flag = False
                        elif att_type[i] == 'R':
                            value = float(value)
                        elif att_type[i] == 'N':
                            value = int(value)
                        elif att_type[i] == 'C':
                            if not attribute_detail[attribute[i]].__contains__(value):
                                attribute_detail[attribute[i]][value] = len(attribute_detail[attribute[i]])
                            value = attribute_detail[attribute[i]][value]
                        example.append(value)
                    if flag:
                        if not equiprobable:
                            examples.append(example)
                        else:
                            if example[-1] == 1:
                                positive_examples.append(example)
                                pos_count += 1
                            else:
                                negative_examples.append(example)
                                neg_count += 1
                    idx = idx + 1
        
        # Data sampling
        if equiprobable:
            if dataset_id in Unbalanced_Data:
                if len(positive_examples) > maxN:
                    data = [i for i in range(pos_count)]
                    df = pd.DataFrame(data= {'examples': data})
                    positive_sampling = [d for d in df.sample(n=maxN, replace = False, random_state = 1)['examples']]
                    positive_sampling = sorted(positive_sampling)
                else:
                    positive_sampling = [i for i in range(pos_count)]
                if len(negative_examples) > maxN:
                    data = [i for i in range(neg_count)]
                    df = pd.DataFrame(data= {'examples': data})
                    negative_sampling = [d for d in df.sample(n=maxN, replace = False, random_state = 1)['examples']]
                    negative_sampling = sorted(negative_sampling)
                else:
                    negative_sampling = [i for i in range(neg_count)]
                '''data = [i for i in range(neg_count)]
                df = pd.DataFrame(data= {'examples': data})
                negative_sampling = [d for d in df.sample(n=maxN, replace = False, random_state = 1)['examples']]
                negative_sampling = sorted(negative_sampling)
                
                print('positive sample: ',pos_count)
                print('negative sample: ',len(negative_sampling))'''
                examples = []
                pos_split = int(0.8 * len(positive_sampling))
                neg_split = int(0.8 * len(negative_sampling))
                for i in range(neg_split):
                    examples.append(negative_examples[negative_sampling[i]])
                for i in range(pos_split):
                    examples.append(positive_examples[positive_sampling[i]])
                for i in range(neg_split,len(negative_sampling)):
                    examples.append(negative_examples[negative_sampling[i]])
                for i in range(pos_split,len(positive_sampling)):
                    examples.append(positive_examples[positive_sampling[i]])
                '''print('sample split: ',pos_split+neg_split)
                print('data split: ',int(0.8* (pos_count+len(negative_sampling))))
                print(examples[neg_split+pos_split-1][-1],examples[pos_split+len(negative_sampling)][-1],examples[neg_split][-1],examples[pos_count+len(negative_sampling)-1][-1])
                print(examples[0][-1],examples[neg_split-1][-1],examples[neg_split+pos_split][-1],examples[len(negative_sampling)+pos_split-1][-1])'''
            else:
                '''print('positive sample: ',pos_count)
                print('negative sample: ',neg_count)'''
                examples = []
                pos_split = int(0.8 * pos_count)
                neg_split = int(0.8 * neg_count)
                for i in range(neg_split):
                    examples.append(negative_examples[i])
                for i in range(pos_split):
                    examples.append(positive_examples[i])
                for i in range(neg_split,neg_count):
                    examples.append(negative_examples[i])
                for i in range(pos_split,pos_count):
                    examples.append(positive_examples[i])
                '''print('sample split: ',pos_split+neg_split)
                print('data split: ',int(0.8* (pos_count+neg_count)))
                print(examples[neg_split+pos_split-1][-1],examples[pos_split+neg_count][-1],examples[neg_split][-1],examples[pos_count+neg_count-1][-1])
                print(examples[0][-1],examples[neg_split-1][-1],examples[neg_split+pos_split][-1],examples[neg_count+pos_split-1][-1])'''
        #Missing Data Handling
        
        
        if takeMiss:
            for idx in range(len(attribute)-1):
                if len(Has_missing[attribute[idx]]) > 0 :
                    probabledData = [0,0]
                    if att_type[idx] != 'C':
                        counter = [0,0]
                        for exp in range(len(examples)):
                            if not Has_missing[attribute[idx]].__contains__(exp):
                                probabledData[examples[exp][-1]] += examples[exp][idx]
                                counter[examples[exp][-1]] +=1
                        for i in range(2):
                            probabledData[i] /= counter[i]
                    else:
                        count = [[],[]]
                        for j in range(2):
                            count[j] = [[0,i] for i in range(len(Has_missing[attribute[idx]]))]
                        for exp in range(len(examples)):
                            if not Has_missing[attribute[idx]].__contains__(exp):
                                count[examples[exp][-1]][examples[exp][idx]][0] += 1
                        for j in range(2):
                            count[j] = sorted(count[j])
                            probabledData[j] = count[j][-1][1]
                    for exp in Has_missing[attribute[idx]]:
                        examples[exp][idx] = probabledData[examples[exp][-1]]
    return examples, attribute_detail

# Decision Tree Implementation

'''
Here ,
    data = all data as NxM matrix format where N = number of samples and M = number of attribute
    attributes = attribute-name
    attribute_types = Types of attribute
    attribute_details = attribute-details (attribute : set of values)
'''
class DecisionTreeClassifier:
    def __init__(self, data, attributes, attribute_types, attribute_details):
        self.data = data
        self.attributes = attributes
        self.attribute_types = attribute_types
        self.attribute_details = attribute_details
        
    '''
        returns which class has maximum frequency
        examples = all examples index
    '''
    
    def Plurality_Value(self,examples):
        count_vector = [0, 0]
        for example in examples:
            count_vector[self.data[example][-1]] = count_vector[self.data[example][-1]] + 1
        if count_vector[0] > count_vector[1] :
            return 0
        else:
            return 1
    
    '''
        Returns True if all examples are in same class, else return False and class if is_same_class = True
        examples = all examples index
    '''
    def Same_Class(self,examples):
        is_same_class = True
        class_name = self.data[examples[0]][-1]
        for example in examples:
            if class_name != self.data[example][-1]:
                is_same_class = False
                break
        return is_same_class, class_name
    
    '''
        returns gain, split condition and subexamples
        attribute = attribute to consider
        examples = examples set
    '''
    
    def binarization_numerical(self,attribute,examples): # Output gain, split condition, subexamples[2]
        labled_value = []
        for exp in examples:
            labled_value.append((self.data[exp][attribute],self.data[exp][-1], exp))
        labled_value = sorted(labled_value)
        positive = []
        negative = []
        split_point = []
        pos_count = 0
        neg_count = 0
        if labled_value[0][1] == 1:
            pos_count = pos_count + 1
        else:
            neg_count = neg_count + 1
        for i in range(1, len(labled_value)):
            if labled_value[i-1][1] != labled_value[i][1] and labled_value[i-1][0] != labled_value[i][0]:
                positive.append(pos_count)
                negative.append(neg_count)
                split_point.append((labled_value[i][0]+labled_value[i-1][0])/2)
            if labled_value[i][1] == 1:
                pos_count = pos_count + 1
            else:
                neg_count = neg_count + 1
        
        max_gain = 0
        split_idx = 0
        for i in range(len(split_point)):
            gain = Gain([positive[i],pos_count-positive[i]],[negative[i],neg_count-negative[i]])
            if gain > max_gain:
                max_gain = gain
                split_idx = i
        if max_gain != 0:
            subexamples = [[],[]]
            for value in labled_value:
                if value[0] < split_point[split_idx]:
                    subexamples[0].append(value[2])
                else:
                    subexamples[1].append(value[2])
            return max_gain, split_point[split_idx], subexamples
        else:
            return 0, None, None
    
    '''
        return attribute for which its gain is maximum, subexamples after branching, max_sp = split point if attribute-type is Numerical or Real
        attributes = attribute-set
        examples = example-set
    '''
    def Importance(self,attributes, examples):
        attribute = attributes[0]
        subexamples = []
        max_gain  = 0
        max_sp = 0
        for a in attributes:
            
            if self.attribute_types[a] == 'C':
                subexps = []
                positive = []
                negative = []
                for i in range(len(self.attribute_details[self.attributes[a]])):
                    positive.append(0)
                    negative.append(0)
                    subexps.append([])
                for exp in examples:
                    subexps[self.data[exp][a]].append(exp)
                    if self.data[exp][-1]== 1:
                        positive[self.data[exp][a]] = positive[self.data[exp][a]] + 1
                    else:
                        negative[self.data[exp][a]] = negative[self.data[exp][a]] + 1
                gain = Gain(positive,negative)
            else:
                gain, split_point, subexps = self.binarization_numerical(a,examples)
            if gain>max_gain:
                max_gain= gain
                attribute = a
                subexamples = subexps
                if self.attribute_types[attribute] != 'C':
                    max_sp = split_point
        if max_gain == 0:
            return None, None, None
        return attribute, subexamples, max_sp
    
    '''
        returns a decision tree
        examples = training examples-set
        atts = attribute-set
        parent_examples = examples-set of it's predecessor
        depth = depth of the tree
    '''
    
    def Decision_Tree_Learning(self,examples, atts, parent_examples, depth):
        if len(examples) == 0:
            tree = Tree()
            tree.answer = self.Plurality_Value(parent_examples)
            return tree
        elif len(atts) == 0:
            tree = Tree()
            tree.answer = self.Plurality_Value(examples)
            return tree
        elif depth == 0:
            tree = Tree()
            tree.answer = self.Plurality_Value(examples)
            return tree
        is_same_class, class_name = self.Same_Class(examples)
        if is_same_class:
            tree = Tree()
            tree.answer = class_name
            return tree
        else:
            attribute,subexamples,split = self.Importance(atts, examples)
            
            
            if attribute is None:
                tree = Tree()
                tree.answer = self.Plurality_Value(examples)
                return tree
            root = Tree()
            root.attribute = attribute
            if self.attribute_types[attribute] == 'C':
                atts.remove(attribute)
            k = 0
            if self.attribute_types[attribute] != 'C':
                tree1 = self.Decision_Tree_Learning(subexamples[0],atts,examples,depth-1)
                tree2 = self.Decision_Tree_Learning(subexamples[1],atts,examples,depth-1)
                tree1.condition = ["<",split]
                tree2.condition = [">=", split]
                root.children["<"] = tree1
                root.children[">="] = tree2
            else:
                for subexp in subexamples:
                    tree = self.Decision_Tree_Learning(subexp, atts,examples,depth - 1)
                    tree.condition = ["=",k]
                    root.children[k] = tree
                    k = k + 1
            return root
            
    '''
        Shows the performance of decision-tree 
        tree = Decision tree
        examples = testing example set
    '''
    
    def Decision_Tree_Testing(self, tree, examples):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for exp in examples:
            root = tree
            while(root.answer == None):
                a = root.attribute
                if self.attribute_types[a] == 'C':
                    root = root.children[self.data[exp][a]]
                else:
                    if self.data[exp][a] < root.children["<"].condition[1]:
                        root = root.children["<"]
                    else:
                        root = root.children[">="]
            if self.data[exp][-1] == 1:
                if root.answer == 1:
                    tp = tp + 1
                else:
                    fn = fn +1
            else:
                if root.answer ==0:
                    tn = tn + 1
                else:
                    fp = fp + 1
        '''print("True Positive: ",tp)
        print("True Negative: ",tn)
        print("False Positive: ",fp)
        print("False Negative: ",fn)'''
        print("Accuracy: ",((tp+tn)*100.0)/(tp+tn+fp+fn))
        print("True positive rate (sensitivity, recall, hit rate): ",tp/(tp+fn))
        print("True negative rate (specificity): ", tn/(tn+fp))
        print("Positive predictive value (precision):  ", tp/(tp+fp))
        print("False discovery rate: ",fp/(fp+tp))
        print("F1 Score: ",(2.0*tp)/(2.0*tp+fp+fn))
        
        
        
        
    # AdaBoost Implementation
    
    '''
        returns result of the examples for the tree
        tree = Decision tree
        examples = testing example set
    '''
    
    def Hypothesis(self,tree,examples):
        answer = [0 for i in range(len(examples))]
        idx = 0
        for exp in examples:
            root = tree
            while(root.answer == None):
                a = root.attribute
                if self.attribute_types[a] == 'C':
                    root = root.children[self.data[exp][a]]
                else:
                    if self.data[exp][a] < root.children["<"].condition[1]:
                        root = root.children["<"]
                    else:
                        root = root.children[">="]
            answer[idx] = root.answer
            idx = idx + 1
        return answer
    
    '''
        Returns,
        h =  decision tree set
        z = weight of the classifier
        
        examples = training example set
        K = number of hypothesis
    '''
    
    def AdaBoost_Learning(self, examples, K):
        N = len(examples)
        w = [1.0/N for i in range(N)]
        h = []
        z = []
        df = pd.DataFrame(data= {'examples': examples})
        atts = [i for i in range(len(self.attributes)-1)]
        for k in range(K):
            #print(k,'-----------------------------------------------------------------------------------')
            data = df.sample(frac=1, replace= True, weights = w, random_state=1)
            sample_examples = [d for d in data['examples']]
            tree = self.Decision_Tree_Learning(sample_examples,atts,[],100)
            answer = self.Hypothesis(tree,examples)
            
            error = 0.0
            for i in range(N):
                if answer[i] != self.data[examples[i]][-1]:
                    error = error + w[examples[i]]
            if error > 0.5:
                continue
            h.append(tree)
            for i in range(N):
                if answer[i] == self.data[examples[i]][-1] and error!=0.0:
                    w[examples[i]] = w[examples[i]] * error/(1-error)
            sumw = 0.0
            for j in range(len(w)):
                sumw = sumw + w[j]
            for j in range(len(w)):
                w[j] = w[j]/sumw
            if error==0.0:
                z.append(float("inf"))
            else:
                z.append(math.log2((1-error)/error))
        return h,z
    
    '''
        Show performance of adaboost implementation
        examples = testing example set
        h =  decision tree set
        z = weight of the classifier
    '''
    def AdaBoost_Testing(self, examples, h, z):
        tp=0
        tn=0
        fp=0
        fn=0
        predict = [0 for i in range(len(examples))]
        for i in range(len(h)):
            answer= self.Hypothesis(h[i], examples)
            for j in range(len(answer)):
                if answer[j] == 1:
                    predict[j] = predict[j] + z[i]
                else:
                    predict[j] = predict[j] - z[i]
        for i in range(len(examples)):
            if predict[i] > 0:
                if self.data[examples[i]][-1] == 1:
                    tp = tp + 1
                else:
                    fp = fp + 1
            else:
                if self.data[examples[i]][-1] == 0:
                    tn = tn + 1
                else:
                    fn = fn + 1
        '''print("True Positive: ",tp)
        print("True Negative: ",tn)
        print("False Positive: ",fp)
        print("False Negative: ",fn)'''
        print("Accuracy: ",((tp+tn)*100.0)/(tp+tn+fp+fn))
        '''print("True positive rate (sensitivity, recall, hit rate): ",tp/(tp+fn))
        print("True negative rate (specificity): ", tn/(tn+fp))
        print("Positive predictive value (precision):  ", tp/(tp+fp))
        print("False discovery rate: ",fp/(fp+tp))
        print("F1 Score: ",(2.0*tp)/(2.0*tp+fp+fn))'''

'''
    Show the result of decision tree or adaboost classification
    idx = 0-telco, 1-creditcardfraud, 2- adult
    takeMiss = True- if missing data should be taken, False- otherwise
    dtcORab = True - decision tree, False - Adaboost, 
    Klist = k -values set
'''

def Decision_Tree_OR_AdaBoost_Two_Approach(idx,takeMiss, dtcORab=True, Klist= None):
    if idx == 0 :
        print('telco-customer-churn   ----------------------------------DATASET')
        Data, Attribute_detail = getData(Files[idx],Dataset_ID[idx],Attribute[idx],Att_Type[idx],{},takeMiss)
        
        dtc = DecisionTreeClassifier(Data,Attribute[idx],Att_Type[idx],Attribute_detail)
        
        atts = [i for i in range(len(Attribute[idx])-1)]
        split = int(0.8 * len(Data))
        train = [i for i in range(0,split+1)]
        test = [i for i in range(split+1,len(Data))]
        
        
    elif idx == 1:
        print('creditcardfraud ------------------------------------ DATASET')
        if dtcORab:
            Data, Attribute_detail = getData(Files[idx],Dataset_ID[idx],Attribute[idx],Att_Type[idx],{}, takeMiss)
        else:
            Data, Attribute_detail = getData(Files[idx],Dataset_ID[idx],Attribute[idx],Att_Type[idx],{}, takeMiss, True, 20000)
        
        dtc = DecisionTreeClassifier(Data,Attribute[idx],Att_Type[idx],Attribute_detail)
        
        atts = [i for i in range(len(Attribute[idx])-1)]
        
        samples = [i for i in range(len(Data))]
        
        split = int(0.8 * len(Data))
        train = [i for i in range(split)]
        test = [i for i in range(split,len(Data))]
    elif idx == 2:
        print('adult --------------------------------------------- DATASET')
        Data_Train, Attribute_detail = getData(Files[idx],Dataset_ID[idx],Attribute[idx],Att_Type[idx],{}, takeMiss)
        
        Data_Test, Attribute_detail = getData(Files[idx+1],Dataset_ID[idx+1],Attribute[idx+1],Att_Type[idx+1],Attribute_detail,takeMiss)
        Data = Data_Train + Data_Test
        
        dtc = DecisionTreeClassifier(Data,Attribute[idx],Att_Type[idx],Attribute_detail)
        
        atts = [i for i in range(len(Attribute[idx])-1)]
        train = [i for i in range(len(Data_Train))]
        test = [len(Data_Train)+i for i in range(len(Data_Test))]
        
    if dtcORab:
        print('-------------------DECISION TREE-------------------')
        tree = dtc.Decision_Tree_Learning(train,atts,[],100)
        
        print('Testing Accuracy ::')
        dtc.Decision_Tree_Testing(tree, test)
        
        print('Training Accuracy ::')
        dtc.Decision_Tree_Testing(tree, train)
    else:
        print('------------------ADABOOST-------------------------')
        
        for k in Klist:
            print('K         :::::::::::::       ',k)
            h, z = dtc.AdaBoost_Learning(train,k)
            
            print('Testing Accuracy :: ')
            dtc.AdaBoost_Testing(test, h, z)
            
            print('Training Accuracy :: ')
            dtc.AdaBoost_Testing(train,h,z)




def Choose_Option(method,idx):
    if method == "DT":
        Decision_Tree_OR_AdaBoost_Two_Approach(idx, True)
    elif method == "AB":
        Decision_Tree_OR_AdaBoost_Two_Approach(idx, True, False, [5,10,15,20])
        
        
def main(argv):
    random.seed(0)
    method = ''
    dataset = ''
    try:
        opts, args = getopt.getopt(argv,"m:d:",["method=","dataset="])
        for opt, arg in opts:
            if opt in ("-m","--method"):
                method = arg
            elif opt in ("-d","--dataset"):
                dataset = int(arg)
        print(method,dataset)
        if method in ['DT','AB'] and dataset in [0,1,2]:
            Choose_Option(method,dataset)
    except getopt.GetoptError:
        print('1505023.py -m <method> -d <dataset>')
        sys.exit(2)

if __name__ == '__main__':
    main(sys.argv[1:])