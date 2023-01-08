import csv
import random
import numpy as np

def read_data(csv_path):
    """Read in the training data from a csv file.
    
    The examples are returned as a list of Python dictionaries, with column names as keys.
    """
    examples = []
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for example in csv_reader:
            for k, v in example.items():
                if v == '':
                    example[k] = None
                else:
                    try:
                        example[k] = float(v)
                    except ValueError:
                         example[k] = v
            examples.append(example)
    return examples


def train_test_split(examples, test_perc):
    """Randomly data set (a list of examples) into a training and test set."""
    test_size = round(test_perc*len(examples))    
    shuffled = random.sample(examples, len(examples))
    return shuffled[test_size:], shuffled[:test_size]

def get_data_dict(examples):
    all_variables = list(examples[0].keys())
    data_dict = dict.fromkeys(all_variables) 
    # create a dictionary that use variable name as key and a list of all values of that variable as value
    for example in examples:
        for variable in example:
            if data_dict[variable] == None:
                data_dict[variable] = [example[variable]]
            else:
                temp = data_dict[variable].copy()
                temp.append(example[variable])
                data_dict[variable] = temp
    return data_dict

def entropy(y):
    # calculate the entropy
    count = []
    for unique_val in set(y):
        # count the number of every unique value
        count.append(y.count(unique_val))
    
    p = np.array(count) / len(y)
    entropy = np.sum(-p*np.log2(p+1e-9))
    return entropy

def information_gain(y, variable, split_choice):
    # function used to calculate information gain
    """
    y: traget variable
    variable: variable list
    split_chioce: value used to split the data
    """
    mask = np.where(np.array(variable)  < split_choice)[0]
    _mask = np.where(np.array(variable) >= split_choice)[0]
    a = len(mask)
    b = len(variable) - a
    
    if(a == 0 or b ==0): 
        ig = 0
        
    y = np.array(y)
    
    ig = entropy(list(y))-a/(a+b)*entropy(list(y[mask]))-b/(a+b)*entropy(list(y[_mask]))
  
    return ig

def best_ig_split(x, y):
    # get the best split value based on given variable
    """
    x: variable list
    y: target
    """
    split_value = []
    ig = []
    
    x = list(filter(None,x))
    
    choices =sorted(set(x))
    
    for choice in choices:
        choice_ig = information_gain(y, x, choice)
        ig.append(choice_ig)
        split_value.append(choice)
    
    # Check if there are more than 1 results if not, return False
    if len(ig) == 0:
        return(None, None, False)

    else:
    # Get results with highest IG
        best_ig = max(ig)
        best_ig_index = ig.index(best_ig)
        best_split = split_value[best_ig_index]
        return(best_ig, best_split, True)

def get_best_split(target_name, id_name, data):
    # loop over all variable to find the best split
    
    variable_names = list(data.keys())
    variable_names.remove(target_name)
    variable_names.remove(id_name)
    split_variable = None
    split_value = None
    ig = 0
    
    for name in variable_names:
        variable_ig, variable_split, _ = best_ig_split(data[name], data[target_name])
        if variable_ig is None:
            pass
        elif variable_ig > ig:
            split_variable = name
            split_value = variable_split
            ig = variable_ig
            
    return (split_variable, split_value, ig)

def make_split(variable, value, examples):
    list_1 = []
    list_2 = []
    for example in examples:
        # missing value handled by left tree
        if example[variable] is None:
            list_1.append(example)
        elif example[variable] < value:
            list_1.append(example)
        else:
            list_2.append(example)
    return (list_1, list_2)

class TreeNodeInterface():
    """Simple "interface" to ensure both types of tree nodes must have a classify() method."""
    def classify(self, example): 
        raise NotImplementedError


class DecisionNode(TreeNodeInterface):
    """Class representing an internal node of a decision tree."""

    def __init__(self, test_attr_name, test_attr_threshold, child_lt, child_ge, miss_lt):
        """Constructor for the decision node.  Assumes attribute values are continuous.

        Args:
            test_attr_name: column name of the attribute being used to split data
            test_attr_threshold: value used for splitting
            child_lt: DecisionNode or LeafNode representing examples with test_attr_name
                values that are less than test_attr_threshold
            child_ge: DecisionNode or LeafNode representing examples with test_attr_name
                values that are greater than or equal to test_attr_threshold
            miss_lt: True if nodes with a missing value for the test attribute should be 
                handled by child_lt, False for child_ge                 
        """    
        self.test_attr_name = test_attr_name  
        self.test_attr_threshold = test_attr_threshold 
        self.child_ge = child_ge
        self.child_lt = child_lt
        self.miss_lt = miss_lt

    def classify(self, example):
        """Classify an example based on its test attribute value.
        
        Args:
            example: a dictionary { attr name -> value } representing a data instance

        Returns: a class label and probability as tuple
        """
        test_val = example[self.test_attr_name]
        if test_val is None:
            child_miss = self.child_lt if self.miss_lt else self.child_ge
            return child_miss.classify(example)
        elif test_val < self.test_attr_threshold:
            return self.child_lt.classify(example)
        else:
            return self.child_ge.classify(example)

    def __str__(self):
        return "test: {} < {:.4f}".format(self.test_attr_name, self.test_attr_threshold)
 

class LeafNode(TreeNodeInterface):
    """Class representing a leaf node of a decision tree.  Holds the predicted class."""

    def __init__(self, pred_class, pred_class_count, total_count):
        """Constructor for the leaf node.

        Args:
            pred_class: class label for the majority class that this leaf represents
            pred_class_count: number of training instances represented by this leaf node
            total_count: the total number of training instances used to build the leaf node
        """    
        self.pred_class = pred_class
        self.pred_class_count = pred_class_count
        self.total_count = total_count
        self.prob = pred_class_count / total_count  # probability of having the class label

    def classify(self, example):
        """Classify an example.
        
        Args:
            example: a dictionary { attr name -> value } representing a data instance

        Returns: a class label and probability as tuple as stored in this leaf node.  This will be
            the same for all examples!
        """
        return self.pred_class, self.prob

    def __str__(self):
        return "leaf {} {}/{}={:.2f}".format(self.pred_class, self.pred_class_count, 
                                             self.total_count, self.prob)
    
class DecisionTree:
    """Class representing a decision tree model."""

    def __init__(self, examples, id_name, class_name, min_leaf_count=1):
        """Constructor for the decision tree model.  Calls learn_tree().

        Args:
            examples: training data to use for tree learning, as a list of dictionaries
            id_name: the name of an identifier attribute (ignored by learn_tree() function)
            class_name: the name of the class label attribute (assumed categorical)
            min_leaf_count: the minimum number of training examples represented at a leaf node
        """
        self.id_name = id_name
        self.class_name = class_name
        self.min_leaf_count = min_leaf_count

        # build the tree!
        self.root = self.learn_tree(examples)  

    def learn_tree(self, examples):
        """Build the decision tree based on entropy and information gain.
        
        Args:
            examples: training data to use for tree learning, as a list of dictionaries.  The
                attribute stored in self.id_name is ignored, and self.class_name is consided
                the class label.
        
        Returns: a DecisionNode or LeafNode representing the tree
        """
        #
        
        # create a dictionary that use variable name as key and a list of all values of that variable as value
        data = get_data_dict(examples)
        
        split_variable, split_value, ig = get_best_split(self.class_name, self.id_name, data)
        left, right = make_split(split_variable, split_value, examples)
        
        left_data = get_data_dict(left)
        right_data = get_data_dict(right)
        
        # if left examples only have 1 class, create a leaf node
        if len(set(left_data[self.class_name])) == 1:
            left_node = LeafNode(left_data[self.class_name][0], len(data), len(data))
        else:
            temp_variable, temp_value, temp_ig = get_best_split(self.class_name, self.id_name, left_data)
            child_left, child_right = make_split(temp_variable, temp_value, left)
            # if any child of it has less number of instances than min_leaf_count, create a lead node
            if len(child_left) < self.min_leaf_count or len(child_right) < self.min_leaf_count:
                # get majority class
                classes = left_data[self.class_name]
                pred_class = max(set(classes), key = classes.count)
                
                left_node = LeafNode(pred_class, classes.count(pred_class) , len(left))
            
            else:
                left_node = self.learn_tree(left)
        
        # if right examples only have 1 class, create a leaf node
        if len(set(right_data[self.class_name])) == 1:
            right_node = LeafNode(right_data[self.class_name][0], len(right), len(data))
        else:
            temp_variable, temp_value, temp_ig = get_best_split(self.class_name, self.id_name, right_data)
            child_left, child_right = make_split(temp_variable, temp_value, right)
            # if any child of it has less number of instances than min_leaf_count, create a lead node
            if len(child_left) < self.min_leaf_count or len(child_right) < self.min_leaf_count:
                # get majority class
                classes = right_data[self.class_name]
                pred_class = max(set(classes), key = classes.count)
                
                right_node = LeafNode(pred_class, classes.count(pred_class) , len(right))
            
            else:
                right_node = self.learn_tree(right)
        
        
        node = DecisionNode(split_variable, split_value, left_node, right_node, True)
            
        return node  # fix this line!
    
    def classify(self, example):
        """Perform inference on a single example.

        Args:
            example: the instance being classified

        Returns: a tuple containing a class label and a probability
        """
        #
        node = self.root
        class_, prob = node.classify(example)
        #
        return class_, prob # fix this line!

    def __str__(self):
        """String representation of tree, calls _ascii_tree()."""
        ln_bef, ln, ln_aft = self._ascii_tree(self.root)
        return "\n".join(ln_bef + [ln] + ln_aft)

    def _ascii_tree(self, node):
        """Super high-tech tree-printing ascii-art madness."""
        indent = 6  # adjust this to decrease or increase width of output 
        if type(node) == LeafNode:
            return [""], "leaf {} {}/{}={:.2f}".format(node.pred_class, node.pred_class_count, node.total_count, node.prob), [""]  
        else:
            child_ln_bef, child_ln, child_ln_aft = self._ascii_tree(node.child_ge)
            lines_before = [ " "*indent*2 + " " + " "*indent + line for line in child_ln_bef ]            
            lines_before.append(" "*indent*2 + u'\u250c' + " >={}----".format(node.test_attr_threshold) + child_ln)
            lines_before.extend([ " "*indent*2 + "|" + " "*indent + line for line in child_ln_aft ])

            line_mid = node.test_attr_name
            
            child_ln_bef, child_ln, child_ln_aft = self._ascii_tree(node.child_lt)
            lines_after = [ " "*indent*2 + "|" + " "*indent + line for line in child_ln_bef ]
            lines_after.append(" "*indent*2 + u'\u2514' + "- <{}----".format(node.test_attr_threshold) + child_ln)
            lines_after.extend([ " "*indent*2 + " " + " "*indent + line for line in child_ln_aft ])

            return lines_before, line_mid, lines_after


def test_model(model, test_examples):
    """Test the tree on the test set and see how we did."""
    correct = 0
    test_act_pred = {}
    for example in test_examples:
        actual = example[model.class_name]
        pred, prob = model.classify(example)
        print("{:30} pred {:15} ({:.2f}), actual {:15} {}".format(example[model.id_name] + ':', 
                                                            "'" + pred + "'", prob, 
                                                            "'" + actual + "'",
                                                            '*' if pred == actual else ''))
        if pred == actual:
            correct += 1
        test_act_pred[(actual, pred)] = test_act_pred.get((actual, pred), 0) + 1 

    acc = correct/len(test_examples)
    return acc, test_act_pred


def confusion2x2(labels, vals):
    """Create an normalized predicted vs. actual confusion matrix for four classes."""
    n = sum([ v for v in vals.values() ])
    abbr = [ "".join(w[0] for w in lab.split()) for lab in labels ]
    s =  ""
    s += " actual _________________  \n"
    for ab, labp in zip(abbr, labels):
        row = [ vals.get((labp, laba), 0)/n for laba in labels ]
        s += "       |        |        | \n"
        s += "  {:^4s} | {:5.2f}  | {:5.2f}  | \n".format(ab, *row)
        s += "       |________|________| \n"
    s += "          {:^4s}     {:^4s} \n".format(*abbr)
    s += "            predicted \n"
    return s

#############################################
