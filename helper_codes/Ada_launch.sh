#!/bin/bash
start_time="$(date +%s%N)"
date
`python NBAdaBoost.py ../dataset/adult.train ../dataset/adult.test > ./ada_result/adult.txt`
date
end_time="$(date +%s)"
runtime="$((end_time-start_time))"
runtime_minutes1="$((runtime/60000000000))"
echo -e "\n"

start_time="$(date +%s%N)"
date
`python NBAdaBoost.py ../dataset/breast_cancer.train ../dataset/breast_cancer.test > ./ada_result/breast_cancer.txt`
date
end_time="$(date +%s%N)"
runtime="$((end_time-start_time))"
runtime_minutes2="$((runtime/60000000000))"
echo -e "\n"

start_time="$(date +%s%N)"
date
`python NBAdaBoost.py ../dataset/led.train ../dataset/led.test > ./ada_result/led.txt`
date
end_time="$(date +%s%N)"
runtime="$((end_time-start_time))"
runtime_minutes3="$((runtime/60000000000))"
echo -e "\n"

start_time="$(date +%s%N)"
date
`python NBAdaBoost.py ../dataset/poker.train ../dataset/poker.test > ./ada_result/poker.txt`
date
end_time="$(date +%s%N)"
runtime="$((end_time-start_time))"
runtime_minutes4="$((runtime/60000000000))"
echo -e "\n"

if [[ ("$runtime_minutes1" > 3 ) || ( "$runtime_minutes2" > 3 ) || ("$runtime_minutes3" > 3 ) || ("$runtime_minutes4" > 3 )]]
then
	echo -e "\n\nTrouble!"
fi

echo -e "Analyze ada_result:\npython Ada_analyze.py"
