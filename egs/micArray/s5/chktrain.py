import os
import sys
import json
import pdb

if __name__ =="__main__":
    savejson="bug_train.json"
    towrite=[]
    p=0
    dp=0
    with open("train.json","rb") as f:
        trainjson=json.load(f)
    for line in open("base.log").readlines():
        line=line.strip()[1:-1].replace("'","").split(", ")
        for utt in line:
            p+=1
            for ind in range(len(trainjson)):
                if utt in trainjson[ind]["utt"]:
                    towrite.append(trainjson[ind])
                    dp+=1
                    break
    if p==dp:
        with open(savejson,"w",encoding="utf8") as fw:
            json.dump(towrite,fw,ensure_ascii=False, indent=2)
        print("Done")
        exit(0)
    else:
        print("p is {}, dp is {}. Not match".format(p,dp))
        exit(1)

