#!/usr/bin/python
'''This program is used to implement the AdaBoost Algorithm with the Naive Bayes Algorithm
	 To run this program, type the following command:
	 python NBAdaBoost.py <training_file> <test file>
'''
import NaiveBayes
import sys
import random
import math
total_rounds = 5

'''This function is used to implement the weighted random sampling and
	 return a random sample which is chosen by the weights assigned to the examples.
'''
def weighted_random_sampling(training_data):
	sample_training_data = []
	cumulative_weights = []
	cumulative_weights.append(float(training_data[0]['weight']))
	for example_num in range(1,len(training_data)):
		value =  float(cumulative_weights[example_num - 1]) + float(training_data[example_num]['weight'])
		if value > 1.0:
			value = 1.0
		cumulative_weights.append(value)
	for sample_index_counter in range(0,len(training_data)):
		random_value = round(random.uniform(0.0,1.0),3)
		min_difference = 1.0
		closest_index = 0
		for cumul_wt_index in range(0,len(training_data)):
			current_diff = abs(random_value - cumulative_weights[cumul_wt_index])
			if current_diff < min_difference:
				min_difference = current_diff
				closest_index = cumul_wt_index
		sample_training_data.append(training_data[closest_index])
	return sample_training_data

'''This is the indicator function that returns 0 if the two values passed to it are
	 equal and 1 otherwise.
'''
def indicator(label_value,predicted_value):
	if label_value == predicted_value:
		return 0.0
	return 1.0


'''This function is used to update the weights of examples based on the error of an example.
	 The weights of the examples in which no mistakes are made are reduced.
'''
def update_weights(training_data,error,predictions):
	num_examples = len(training_data)
	for example_index in range(0,num_examples):
		if indicator(training_data[example_index]['label'],predictions[example_index]) == 0.0:
			training_data[example_index]['weight'] = float(training_data[example_index]['weight']) * float(error)/float(1 - error)
	return training_data
		

'''This function is used to find the error of the classifier on the training data given the
	 predictions and actual labels of that data.
'''
def find_error(training_data,predictions):
	error = 0.0
	num_examples = len(training_data)
	for example_index in range(0,num_examples):
		indicator_factor = indicator(training_data[example_index]['label'],predictions[example_index])
		error = float(error) + (float(training_data[example_index]['weight']) * float(indicator_factor))
	return error


'''This function is used to normalize the weights of the different examples.
	 Thus, effectively, the weights of examples on which we make mistakes are more 
	 than the ones on which we classify correctly.
'''
def normalize_weights(training_data):
	weight_sum = 0.0
	for example in training_data:
		weight_sum = float(weight_sum) + float(example['weight'])
	for example in training_data:
		example['weight'] = float(example['weight'])/float(weight_sum)
	return training_data


'''This function is used to calculate the weight that should be assigned to each
	 weak learner (naive Bayes classifier in our case) such that more weight 
	 is given to more accurate learners.
'''
def calculate_weight_classifiers(round_error):
	round_alpha = []
	for error in round_error:
		term = float(1 - error)/float(error)
		alpha = float(math.log(term,2))
		round_alpha.append(alpha)
	return round_alpha
	

'''This is a function that is a wrapper for running the AdaBoost Algorithm.
'''
def run_adaboost(training_data,testing_data,values_in_features,max_index):
	round_error = []
	round_model = []
	round_alpha = []
	for example in training_data:
		example['weight'] = float(1)/float(len(training_data))
	current_round_ctr = 0
	while current_round_ctr < total_rounds:
		sample_training_data = weighted_random_sampling(training_data)
		#print "Done sampling."
		conditional_prob_model= NaiveBayes.train_naive_bayes_get_classifier(sample_training_data,values_in_features)
		#print "Done training on naive bayes model."														#Get the model as got by training on the random sample.
		round_model.append(conditional_prob_model)
		predictions = NaiveBayes.get_predictions_from_model(conditional_prob_model,training_data,max_index)
		error = find_error(training_data,predictions)													#Find the error of the model.
		round_error.append(error)
		#print "Done finding predictions and getting error."
		#print error
		if error >= 0.5:
			break	
		training_data = update_weights(training_data,error,predictions)
		#print "Done updating weights."
		training_data = normalize_weights(training_data)
		#print "Done normalizing weights."
		current_round_ctr += 1
	#print "Done training models for multiple rounds."
	round_alpha = calculate_weight_classifiers(round_error)
	total_classifiers_generated = len(round_error)
	#print "Done calculating alpha."
	adaboost_predictions = [] 																															#Get the boosted predictions for different examples.
	for example in testing_data:
		boosted_prediction = 1
		for current_round_ctr in range(0,total_classifiers_generated):
			predicted_label = "-1"
			features_prob_product_positive = 1.0
			features_prob_product_negative = 1.0
			for feature in range(1,max_index + 1):
				if feature in example:
					pass_value = example[feature]
				else:
					pass_value = 0
				string_lookup = str(feature) + ':' + str(pass_value) + ':' + "+1"
				features_prob_product_positive = float(features_prob_product_positive) * float(round_model[current_round_ctr][string_lookup])
				string_lookup = str(feature) + ':' + str(pass_value) + ':' + "-1"
				features_prob_product_negative = float(features_prob_product_negative) * float(round_model[current_round_ctr][string_lookup])
			if (float(features_prob_product_positive*round_model[current_round_ctr]['prior_positive']) >= 
					float(features_prob_product_negative*round_model[current_round_ctr]['prior_negative'])):
				predicted_label = "+1"
			boosted_prediction = float(boosted_prediction) + float(float(round_alpha[current_round_ctr]) * float(predicted_label))
		if boosted_prediction > 0:
			final_prediction = "+1"
		else:
			final_prediction = "-1"
		adaboost_predictions.append(final_prediction)
	#print "Done with Adaboost predictions."
	return adaboost_predictions

		
if __name__ == "__main__":
	if(len(sys.argv)) != 3:
		print NaiveBayes.usage("NBAdaBoost.py")
		sys.exit(1)
	else:
		train_file_name = sys.argv[1]
		test_file_name = sys.argv[2]
		train_file = open(train_file_name,"r")
		test_file = open(test_file_name,"r")
		training_data,testing_data,values_in_features,max_index = NaiveBayes.process_files(train_file,test_file)
		train_file.close()
		test_file.close()
		adaboost_predictions = run_adaboost(training_data,training_data,values_in_features,max_index)
		NaiveBayes.print_metrics(training_data,adaboost_predictions)
		adaboost_predictions = run_adaboost(training_data,testing_data,values_in_features,max_index)
		NaiveBayes.print_metrics(testing_data,adaboost_predictions)
