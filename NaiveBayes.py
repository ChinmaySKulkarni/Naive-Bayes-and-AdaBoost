#!/usr/bin/python
'''
	This program is used to implement the Naive Bayes Algorithm for classification.
	To run the program, type the following command:
	
	python NaiveBayes.py <training_file> <test_file> 
'''
import sys


'''This function mentions the correct usage for running the program.
'''
def usage(program_name):
	return "Wrong Usage!\nCorrect Usage is:\t<python "+ program_name + "> <train_file> <test_file>"


'''This function is used to parse the given training and testing files and to store the
	 training data set and the testing data set. These data sets are stored as a list of 
	 dictionaries where the label and attribute names (indices) are the keys and their 
	 corresponding values are the values of the dictionaries.
'''
def parse_file(fd,data_list,max_index):
	for line in fd:
		if line.strip():
			line = line.split()
			label_feature_vector = {}
			label_feature_vector['label'] = line[0]
			for feature_value in line[1:]:
				feature_value = feature_value.split(':')
				index = int(feature_value[0])
				value = int(feature_value[1])
				label_feature_vector[index] = value
			if(index > max_index):
				max_index = index																												#Find the maximum attribute index.
			data_list.append(label_feature_vector)
	return max_index,data_list


'''This function is used to find all the distinct values that each attribute can take.
	 This is stored in a dictionary with keys as the attribute names and the value for a 
	 key as a list of distinct values that the attribute can take.
'''
def find_distinct_values_feature(training_data,testing_data,max_index):
	values_in_features = {}
	total_data = training_data + testing_data
	for index in range(1,max_index + 1):
		zero_value_flag = 0
		distinct_values = set()
		for example in total_data:
			if index in example:
				distinct_values.add(example[index])
			else:
				zero_value_flag = 1
		if zero_value_flag == 1:
			distinct_values.add(0)
		values_in_features[index] = distinct_values
	return values_in_features
		

'''This function is the wrapper function which parses the training file and testing files and 
	 calls the find_distinct_values_feature.
'''
def process_files(train_fd,test_fd):
	training_data = []
	testing_data = []
	max_index,training_data = parse_file(train_fd,training_data,0)
	max_index,testing_data = parse_file(test_fd,testing_data,max_index)
	#print "Done storing the training and testing examples as feature vectors."
	values_in_features = find_distinct_values_feature(training_data,testing_data,max_index)
	#print "Done extracting feature information."
	return training_data,testing_data,values_in_features,max_index


'''This function is used to calculate the prior probabilities of the class labels.
'''
def find_prior_probability(label_value,training_data):
	count = 0
	for example in training_data:
		if example['label'] == label_value:
			count += 1
	return float(count)/float(len(training_data))

	
'''This function is basically the model that is learned in the Naive Bayes Classifier.
	 It stores the conditional probability values of each attribute_name -> value -> label
	 combination. These values are stored in a dictionary and looked up using a 
	 string lookup.
'''
def store_all_feature_value_label_cond_probabilities(training_data,values_in_features):
	value_cond_prob = {}
	labels = ["+1","-1"]
	for feature in values_in_features:
		distinct_values = values_in_features[feature]
		total_values_feature = len(distinct_values)
		for value in distinct_values:
			for label_val in labels:
				string_lookup = str(feature) + ':' + str(value) + ':' + label_val
				counter = 0
				total_counter = 0
				for example in training_data:
					if example['label'] == label_val:
						total_counter += 1
						if feature in example:
							if example[feature] == value:
								counter += 1
						else:
							if value == 0: 
								counter += 1
				if counter == 0:
					counter = 1																													#Laplacian Correction.
					total_counter += total_values_feature
				probability =  float(counter)/float(total_counter)
				value_cond_prob[string_lookup] = probability
	return value_cond_prob


'''This function is used for training the Naive Bayes classifier and returning the corresponding 
	 conditional probability values which is the model that is learned.
'''
def train_naive_bayes_get_classifier(training_data,values_in_features):
	prior_positive = find_prior_probability("+1",training_data)
	prior_negative = find_prior_probability("-1",training_data)
	#print "Done finding prior probabilities for class labels."
	value_cond_prob = store_all_feature_value_label_cond_probabilities(training_data,values_in_features)
	value_cond_prob['prior_positive'] = prior_positive
	value_cond_prob['prior_negative'] = prior_negative
	#print "Done storing conditional probabilities for attribute values."
	return value_cond_prob																											#Return the model for the Naive Bayes classifier.


'''This function is used to return the predictions of the classifier on testing data.
'''
def get_predictions_from_model(value_cond_prob,testing_data,max_index):
	predictions = []
	for example in testing_data:
		predicted_label = "-1"
		features_prob_product_positive = 1.0
		features_prob_product_negative = 1.0
		for feature in range(1,max_index + 1):
			if feature in example:
				pass_value = example[feature]
			else:
				pass_value = 0
			string_lookup = str(feature) + ':' + str(pass_value) + ':' + "+1"
			features_prob_product_positive = float(features_prob_product_positive) * float(value_cond_prob[string_lookup])
			string_lookup = str(feature) + ':' + str(pass_value) + ':' + "-1"
			features_prob_product_negative = float(features_prob_product_negative) * float(value_cond_prob[string_lookup])
		if (float(features_prob_product_positive * value_cond_prob['prior_positive']) >= float(features_prob_product_negative * value_cond_prob['prior_negative'])):
			predicted_label = "+1"
		predictions.append(predicted_label)
	return predictions


'''This function is used to evaluate the accuracy/quality of the classifier on the test data
	 and for printing the metrics like the true positives, negatives, etc.
'''
def print_metrics(testing_data,predictions):
	true_positives = 0
	false_negatives = 0
	false_positives = 0
	true_negatives = 0
	num_examples = len(testing_data)
	for example_num in range(0,num_examples):
		predicted_label = predictions[example_num]
		if testing_data[example_num]['label'] == "+1":
			if predicted_label == "+1":
				true_positives += 1
			elif predicted_label == "-1":
				false_negatives += 1
		elif testing_data[example_num]['label'] == "-1":
			if predicted_label == "+1":
				false_positives += 1
			elif predicted_label == "-1":
				true_negatives += 1
	print true_positives,"\t",false_negatives,"\t",false_positives,"\t",true_negatives


if __name__ == "__main__":
	if(len(sys.argv)) != 3:
		print usage("NaiveBayes.py")
		sys.exit(1)
	else:
		train_file_name = sys.argv[1]
		test_file_name = sys.argv[2]
		train_file = open(train_file_name,"r")
		test_file = open(test_file_name,"r")
		training_data,testing_data,values_in_features,max_index = process_files(train_file,test_file)
		train_file.close()
		test_file.close()
		value_cond_prob = train_naive_bayes_get_classifier(training_data,values_in_features)
		predictions = get_predictions_from_model(value_cond_prob,training_data,max_index)
		print_metrics(training_data,predictions)
		predictions = get_predictions_from_model(value_cond_prob,testing_data,max_index)
		print_metrics(testing_data,predictions)
