#!/usr/bin/env python

# parse starfile get particle count from each micrograph
# get reported DF

import sys
import numpy as np
import matplotlib.pyplot as plt
###---------function: read the star file get the header, labels, and data -------------#######
def read_starfile_new(f):
    inhead = True
    alldata = open(f,'r').readlines()
    labelsdic = {}
    data = []
    header = []
    count = 0
    labcount = 0
    for i in alldata:
        if '_rln' in i:
            labelsdic[i.split()[0]] = labcount
            labcount +=1
        if inhead == True:
            header.append(i.strip("\n"))
            if '_rln' in i and '#' in i and  '_rln' not in alldata[count+1] and '#' not in alldata[count+1]:
                inhead = False
        elif len(i.split())>=1:
            data.append(i.split())
        count +=1
    
    return(labelsdic,header,data)
#---------------------------------------------------------------------------------------------#

def parse_starfile(starfile):
    '''for each micrograph [count,defocus]'''
    (labels,header,data) = read_starfile_new(starfile)
    micrographs = {}                #{micrographname: [nparts,defocus,[loglikelyhood values],[maxvalueprobdist values]]}
    for i in data:
        micname = i[labels['_rlnMicrographName']].split('/')[-1]
        try:
            micrographs[micname][0] +=1
        except:
            defocus = (float(i[labels['_rlnDefocusU']])+float(i[labels['_rlnDefocusV']]))/2
            micrographs[micname] = [1,defocus]
        if '_rlnLogLikeliContribution' in labels:
            try:
                micrographs[micname][2].append(float(i[labels['_rlnLogLikeliContribution']]))
            except:
                micrographs[micname].append([float(i[labels['_rlnLogLikeliContribution']])])
            try:
                micrographs[micname][3].append(float(i[labels['_rlnMaxValueProbDistribution']]))
            except:
                micrographs[micname].append([float(i[labels['_rlnMaxValueProbDistribution']])])
    return(micrographs)

errormsg = './usage compare_recon_vs_extract <reconstrction data starfile> <original particles starfile>'

try:
    reconstruction = (parse_starfile(sys.argv[2]))
    initial = (parse_starfile(sys.argv[1]))
except:
    sys.exit(errormsg)

print('Raw dataset:     {0}'.format(len(initial)))
print('Reconstruction:  {0}'.format(len(reconstruction)))

ratios = []
icounts = []
fcounts = []
LLmeans,MVPDmeans,DFs =  [],[],[]
print('micrograph,nparts,defocus,meanLLscore,meanMVPDscore')
for i in initial:
    try:
        print(i,initial[i][0],initial[i][1],reconstruction[i][0],float(reconstruction[i][0])/float(initial[i][0]),np.mean(reconstruction[i][2]),np.mean(reconstruction[i][3]))
        DFs.append(initial[i][1])
        LLmeans.append(np.mean(reconstruction[i][2]))
        MVPDmeans.append(np.mean(reconstruction[i][3]))
        ratios.append(float(reconstruction[i][0])/float(initial[i][0]))
        icounts.append(initial[i][0])
        fcounts.append(reconstruction[i][0])
    except:
        pass

LLmeans = [x/max(LLmeans) for x in LLmeans]
MVPDmeans = [x/max(MVPDmeans) for x in MVPDmeans]

print('-- data overview -- ')
print('parts/micrograph initial: {0}'.format(np.mean(icounts)))
print('parts/micrograph final:   {0}'.format(np.mean(fcounts)))
print('mean initial/final per micrograph:  {0}'.format(np.mean(ratios)))

## diagnostic plots - fit lines to these at some point
## defocus dependence of LL, MVPD, and ratio
f, (ax1, ax2,ax3) = plt.subplots(3, 1,sharex=True, sharey=True)
ax1.scatter(DFs,LLmeans,c='red',label='LogLikelyhood')
ax2.scatter(DFs,MVPDmeans,c='blue',label='MaxValProbDist')
ax3.scatter(DFs,ratios,c='g',label='picked/used')

ax1.legend(loc='lower right')
ax2.legend(loc='best')
ax3.legend(loc='upper right')
plt.xlabel('Defocus')
plt.ylabel('Score')
plt.show()
plt.close()

## Ratio dependence of LL or MVPD
f, (ax1, ax2) = plt.subplots(2, 1,sharex=True, sharey=True)
ax1.scatter(ratios,LLmeans,c='red',label='LogLikelyhood')
ax2.scatter(ratios,MVPDmeans,c='blue',label='MaxValProbDist')

ax1.legend(loc='lower right')
ax2.legend(loc='best')

plt.xlabel('picked/used')
plt.ylabel('score')
plt.show()
plt.close()

## Ratio dependence of LL or MVPD
f, (ax1) = plt.subplots(1, 1,sharex=True, sharey=True)
ax1.scatter(icounts,ratios,c='red')
plt.xlabel('# initially picked')
plt.ylabel('picked/used')
plt.show()
plt.close()

## LL vs MVPD
f, (ax1) = plt.subplots(1, 1,sharex=True, sharey=True)
ax1.scatter(LLmeans,MVPDmeans,c='red')
plt.xlabel('LogLikleyhood')
plt.ylabel('MaxValProbDist')
plt.show()
plt.close()

## histogram of ratios
bins = [float(x)/100 for x in range(0,101,5)]
plt.hist(ratios,bins=bins)
plt.xlabel('picked/used')
plt.ylabel('# micrographs')
plt.show()
plt.close()

