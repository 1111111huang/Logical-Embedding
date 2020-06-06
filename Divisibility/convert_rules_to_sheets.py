import csv
filename='./100/rules'
with open(filename+'.txt','r+') as file:
	rules=file.readlines()
	results=[]
	for rule in rules:
		temp=rule.split()
		score=float(temp[0])
		norm_score=float(temp[1][1:-1])
		rule=''
		for i in temp[2:]:
			rule+=i
		results+=[[score,norm_score,rule]]
with open(filename+'.csv','w+') as file:
	writer = csv.writer(file)
	for result in results:
		writer.writerow(result)
file.close()