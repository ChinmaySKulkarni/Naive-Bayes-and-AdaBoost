#!/usr/bin/python 
#file_names=["./result/adult.txt","./result/breast_cancer.txt","./result/led.txt","./result/poker.txt"]
file_names=["./ada_result/adult.txt","./ada_result/breast_cancer.txt","./ada_result/led.txt","./ada_result/poker.txt"]

def f_beta_score(beta,precision,recall):
	f_score = float((1 + float(beta)*float(beta))*precision*recall)/float((float(beta)*float(beta)*precision) + recall)
	print "f_" + str(beta) + "_score\t",f_score

for name in file_names:
	fd = open(name,"r")
	print name + " :"
	flag = 0
	for line in fd:
		line = line.strip()
		line = line.split("\t")
		tp = int(line[0])
		fn = int(line[1])
		fp = int(line[2])
		tn = int(line[3])		
		if flag == 0:
			print "Training Data:"
			flag += 1
		else:
			print "\nTest Data:"
		accuracy = float(tp + tn)/float(tp + fn + fp + tn)
		print "accuracy\t",accuracy
		error_rate = float(1) - float(accuracy)	
		print "error_rate\t",error_rate
		sensitivity = float(tp)/float(tp + fn)
		print "sensitivity\t",sensitivity
		specificity = float(tn)/float(fp + tn)
		print "specificity\t",specificity
		precision = float(tp)/float(tp + fp)
		print "precision\t",precision
		recall = float(tp)/float(tp + fn)
		print "recall\t\t",recall
		f_beta_score(1,precision,recall)
		f_beta_score(0.5,precision,recall)
		f_beta_score(2,precision,recall)
	print "\n\n"
	fd.close()
