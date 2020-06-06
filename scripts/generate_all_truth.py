upper=100
divides=[]
dividedBy=[]
notRelated=[]
for i in range(1,upper+1):
    for j in range(i,upper+1):
        if j%i==0:
            divides+=["divides "+str(i)+" "+str(j)+"\n"]
            dividedBy+=["dividedBy "+str(j)+" "+str(i)+"\n"]
        else:
            notRelated+=["notRelated "+str(i)+" "+str(j)+"\n"]
            notRelated+=["notRelated "+str(j)+" "+str(i)+"\n"]
with open("divisibility_100_truth.txt","w+") as file:
    for truth in divides:
        file.write(truth)
    for truth in dividedBy:
        file.write(truth)
    for truth in notRelated:
        file.write(truth)
