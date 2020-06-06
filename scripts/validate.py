exp_category="Divisibility"
exp_name="100"
truth_file=exp_category+"_"+exp_name+"_truth.txt"
infered_file="all_infered_"+exp_category+"_"+exp_name+".txt"
truth=[]
with open(truth_file) as file:
    lines=file.readlines()
    for line in lines:
        truth+=[line.split(' ')]
        truth[-1][-1]=truth[-1][-1][:-1]
infer=[]
with open(infered_file) as file:
    lines=file.readlines()
    for line in lines:
        infer+=[line.split(' ')]
        infer[-1][-1]=infer[-1][-1][:-1]
correct_infer=0
for i in infer:
    if i in truth:
        correct_infer+=1
print("correct infer: ", correct_infer, "incorrect infer: ", len(infer)-correct_infer,"not infered: ",len(truth)-correct_infer)
print("total infered results: ", len(infer), "total truths: ", len(truth))
