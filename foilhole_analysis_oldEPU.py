#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import mrcfile
import glob
import xml.etree.ElementTree as ET
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')



# identify all foil holes that had images taken from them
# score each foil hole buy how many particles were picked from it.
# extract each as sub imagesparticles
#

def get_dirs(datapath):
    try:
        metadata_path = os.path.abspath('{0}/Metadata/'.format(datapath))
    except:
        sys.exit('metadata tfiles not found at {0}'.format('{0}/Metadata/'.format(datapath)))
    try:
        GSimages_path = os.path.abspath('{0}/Images-Disc1/'.format(datapath))
    except:
        sys.exit('grid squares not found at {0}'.format('{0}/Images-Disc1/'.format(datapath)))
    return(metadata_path,GSimages_path)

def get_files(dirpath,ext):
    files = glob.glob('{0}/{1}'.format(dirpath,ext))
    print('targeting metadata : {0}/{1}'.format(dirpath,ext))
    print('{0} targets found'.format(len(files)))
    return(files)

def make_bg(square_level_image): 
    '''make the plot with the mrc ofthe gridsquare as background'''
    gridsquare_image = mrcfile.open(square_level_image)
    micdata = gridsquare_image.data
    plt.axis('off')
    plt.imshow(micdata,cmap='Greys_r')
    mic_ed = gridsquare_image.extended_header
    sq_cent_x,sq_cent_y,sq_z,sq_apix = (mic_ed[0][12],mic_ed[0][13],mic_ed[0][14],mic_ed[0][17])
    gridsquare_image.close()
    return(sq_cent_x,sq_cent_y,sq_z,sq_apix)

def extract_square(mrc_array,sqsize,centerx,centery):
    centerx,centery = int(centerx-1),int(centery-1)
    d = int(sqsize/2)
    sq = mrc_array[centery-d:centery+d,centerx-d:centerx+d]
    return(np.mean(sq),np.std(sq))

def parse_xml_GS(xml_file):
    root = ET.parse(xml_file).getroot()
    targets = []
    for i in root.findall(".//"):
        if i.tag.split('}')[-1] == 'Id':
            sqID = i.text
        if i.tag.split('}')[-1] == 'Order':
            order = i.text
        if i.tag.split('}')[-1] == 'X':
            stageX = i.text
        if i.tag.split('}')[-1] == 'Y':
            stageY = i.text
        if i.tag.split('}')[-1] == 'Selected': 
            selected = i.text
        if i.tag.split('}')[-1] =='State':
            completed = i.text
        if i.tag.split('}')[-1] =='FileName':
            targets.append(i.text)
    return(sqID,order,stageX,stageY,selected,completed,targets)
    
def parse_xml_target(xml_file):
    root = ET.parse(xml_file).getroot()
    for i in root.findall(".//"):
        if i.tag == '{http://schemas.datacontract.org/2004/07/Fei.Applications.Common.Types}Id':
            targetID = i.text
        if i.tag.split('}')[-1] == 'Order':
            order = i.text
        if i.tag == '{http://schemas.datacontract.org/2004/07/Fei.SharedObjects}X':
            stageX = i.text
        if i.tag == '{http://schemas.datacontract.org/2004/07/Fei.SharedObjects}Y':
            stageY = i.text
        if i.tag == '{http://schemas.datacontract.org/2004/07/Fei.SharedObjects}Z':
            stageZ = i.text
        if i.tag =='{http://schemas.datacontract.org/2004/07/System.Drawing}x':
            drawX = i.text
        if i.tag =='{http://schemas.datacontract.org/2004/07/System.Drawing}y':
            drawY = i.text
    return(targetID,order,stageX,stageY,stageZ,drawX,drawY)

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


def make_correlation_plot(vals,names):
    print(names)
    if len(vals)==3:
        top3rd1,middle3rd1,bottom3rd1,top3rd2,middle3rd2,bottom3rd2,top3rd3,middle3rd3,bottom3rd3 = [],[],[],[],[],[],[],[],[]
        n=0
        vrange = np.std(vals[2])
        mean = np.mean(vals[2])
        for i in vals[2]:
            if i < mean-vrange:
                bottom3rd1.append(vals[0][n])
                bottom3rd2.append(vals[1][n])
                bottom3rd3.append(vals[2][n])
            elif i>= mean-vrange and i<= mean + vrange:
                middle3rd1.append(vals[0][n])
                middle3rd2.append(vals[1][n])
                middle3rd3.append(vals[2][n])
            elif i > mean + vrange:
                top3rd1.append(vals[0][n])
                top3rd2.append(vals[1][n])
                top3rd3.append(vals[2][n])

            n+=1
        fig = plt.figure()
        main = plt.subplot2grid((3,5),(0,0),rowspan=3, colspan=3)
        main.set_yticklabels([])
        main.set_xticklabels([])
        main.set_xlim((0,imdim[0]))
        main.set_ylim((0,imdim[0]))
        h = main.scatter(vals[0],vals[1],c=vals[2],s=5,edgecolor='face')
        t3rd = plt.subplot2grid((3,5),(0,3))
        t3rd.scatter(top3rd1,top3rd2,c=top3rd3,vmax=max(vals[2]),vmin=min(vals[2]),s=2,edgecolor='face')
        t3rd.set_yticklabels([])
        t3rd.set_xticklabels([])
        t3rd.set_ylim((0,imdim[0]))
        t3rd.set_xlim((0,imdim[0]))
        m3rd = plt.subplot2grid((3,5),(1,3))
        m3rd.scatter(middle3rd1,middle3rd2,c=middle3rd3,vmax=max(vals[2]),vmin=min(vals[2]),s=2,edgecolor='face')
        m3rd.set_yticklabels([])
        m3rd.set_xticklabels([])
        m3rd.set_ylim((0,imdim[0]))
        m3rd.set_xlim((0,imdim[0]))
        b3rd = plt.subplot2grid((3,5),(2,3))
        b3rd.scatter(bottom3rd1,bottom3rd2,c=bottom3rd3,vmax=max(vals[2]),vmin=min(vals[2]),s=2,edgecolor='face')
        b3rd.set_yticklabels([])
        b3rd.set_xticklabels([])
        b3rd.set_ylim((0,imdim[0]))
        b3rd.set_xlim((0,imdim[0]))
        colbar = plt.subplot2grid((3,5),(0,4),rowspan=3)
        fig.colorbar(h,cax=colbar)
        plt.xlabel(names[0])
        plt.ylabel(names[1])
        fig.savefig('{0}_{1}_{2}.png'.format(names[0],names[1],names[2]))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.close()
    else:
        plt.rc('xtick',labelsize=8)
        plt.rc('ytick',labelsize=8)
        fig = plt.figure()
        ax1 = plt.subplot2grid((2,3),(0,0),rowspan=2,colspan=2)
        ax1.scatter(vals[0],vals[1],s=5)
        ax1.set_xlabel(names[0])
        ax1.set_ylabel(names[1])
        ax2 = plt.subplot2grid((2,3),(0,2))
        ax2.hist([x for x in vals[0] if str(x) != 'nan'])
        ax2.set_xlabel(names[0],size=10)
        ax3 = plt.subplot2grid((2,3),(1,2))
        ax3.hist([x for x in vals[1] if str(x) != 'nan'])
        ax3.set_xlabel(names[1])
        fig.savefig('{0}_{1}.png'.format(names[0],names[1]))
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.close()
#---------------------------------------------------------------------------------------------#


# read the starfile - get the number of particles per foil hole
try:
    (labels,header,data) = read_starfile_new(sys.argv[2])
    print('\nreading starfile: {0}'.format(sys.argv[2]))
except:
    sys.exit('ERROR: problem reading starfile {0}'.format(sys.argv[2]))

foilhole_part_count = {}            # {foilholeID:[particle_count ,mean LLC, mean MVPD,meanDefocus,collectiondate/time]}
for line in data:
    foilhole = line[labels['_rlnMicrographName']].split('/')[-1].split('_')[1]
    try:
        foilhole_part_count[foilhole][0] +=1
        foilhole_part_count[foilhole][1].append(float(line[labels['_rlnLogLikeliContribution']]))
        foilhole_part_count[foilhole][2].append(float(line[labels['_rlnMaxValueProbDistribution']]))
        foilhole_part_count[foilhole][3].append(float(line[labels['_rlnDefocusU']]))
        foilhole_part_count[foilhole][3].append(float(line[labels['_rlnDefocusV']]))

    except:
        datetime = ''.join(line[labels['_rlnMicrographName']].split('/')[-1].split('.')[0].split('_')[-3:-1])
        foilhole_part_count[foilhole]= [1,[float(line[labels['_rlnLogLikeliContribution']])],[float(line[labels['_rlnMaxValueProbDistribution']])],[float(line[labels['_rlnDefocusU']]),float(line[labels['_rlnDefocusV']])],datetime]
for i in foilhole_part_count:
    foilhole_part_count[i][1] = np.mean(foilhole_part_count[i][1])
    foilhole_part_count[i][2] = np.mean(foilhole_part_count[i][2])
    foilhole_part_count[i][3] = np.mean(foilhole_part_count[i][3])

print('{0} foilholes  {1} particles'.format(len(foilhole_part_count),len(data)))

# get grid square metadata
print('\nFinding all gridsquares')
metadata,images = get_dirs(sys.argv[1])
GS_dic = {}                 #{gridsquare:[order,stageX,stageY,selected,completed,[targets]]}
GS_metadata= get_files(metadata,'*.dm')
for i in GS_metadata:
    GS_dic[i] = parse_xml_GS(i)

# identify the selected squares - do all the counts and pit the stats for each foil hole in the big ass dictionary

big_GS_dic = {}         # {gridsquare:[{foilhole1:[x,y,nparts,meanLLC,meanMVPD,aqorder]},{foilhole2:[x,y,nparts,meanLLC,meanMVPD,aqorder]}}

selected= []
for i in GS_dic:
    if GS_dic[i][4] == 'true':
        selected.append(i)
print('Found {0} selected gridsquares to process'.format(len(selected)))
GS_imdic = {}			#{GS imagename -just time and data: GS target_number}
for i in selected:
    GS_name = i.split('/')[-1].split('.')[0]
    imagepath = '{0}/{1}/'.format(images,GS_name)
    FHpath = imagepath+'Foilholes'
    print('----------')
    print(GS_name)
    print('gridsquare metadata: {0}'.format(i))
    print('gridsquare image path  : {0}'.format(imagepath))
    print('foilholes path: {0}'.format(FHpath))
    GS_images = glob.glob('{0}/*.mrc'.format(imagepath))
    GS_images.sort()
    GS_image = GS_images[-1]
    big_GS_dic[GS_image] = {}
    print('{0} gridsquare images found :: using {1}'.format(len(GS_images),GS_image.split('/')[-1]))
    sq_cent_x,sq_cent_y,sq_z,sq_apix = make_bg(GS_image)
    GS_imdic[GS_image.split('/')[-1].split('.')[0]] = GS_name
# get targets for the square xy positions from metadata
    tmd_dir = '{0}/{1}'.format(metadata,GS_name)
    target_metadata = get_files(tmd_dir,'*.dm')
    targetsx,targetsy,targetsorder = [],[],[]
    targets_dic = {}            #{targetID:[order,stageX,stageY,stageZ,drawX,drawY]}
    
    # get coords of targets from metadata
    for target in target_metadata:
        targetID,order,stageX,stageY,stageZ,drawX,drawY = parse_xml_target(target)    
        targets_dic[target.split('/')[-1].split('_')[1].replace('.dm','')] = [order,stageX,stageY,stageZ,drawX,drawY]
    
    # mark each foil hole with the number of particles contributed and any misses
    hitsx,hitsy,LLCvals,MVPDvals = [],[],[],[]
    for t in targets_dic:
        if t in foilhole_part_count:
            hitsx.append(float(targets_dic[t][4]))
            hitsy.append(float(targets_dic[t][5]))
            LLCvals.append(foilhole_part_count[t][1])
            MVPDvals.append(foilhole_part_count[t][2])
            big_GS_dic[GS_image][t]=[float(targets_dic[t][4]),float(targets_dic[t][5]),foilhole_part_count[t][0],foilhole_part_count[t][1],foilhole_part_count[t][2],foilhole_part_count[t][3],foilhole_part_count[t][4]]
    missesx,missesy = [],[]
    for shothole in glob.glob('{0}/*.mrc'.format(FHpath)):
        hole = shothole.split('_')[1]
        if hole not in foil_part_count:
            print(hole,'MISS')
            missesx.append(targets_dic[hole][4],targets_dic[hole][5])
            big_GS_dic[GS_image][t]=[float(targets_dic[t][4]),float(targets_dic[t][5]),0,0,0,foilhole_part_count[t][3]]
    plt.scatter(missesx,missesy,c='blue',s=30,alpha=0.8,edgecolors='face')
    plt.scatter(hitsx,hitsy,s=30,alpha=0.8,edgecolors='face',c='orange')
    for t in targets_dic:
        plt.text(float(targets_dic[t][4]),float(targets_dic[t][5]),t,size=3)
        if t in foilhole_part_count:
            plt.text(float(targets_dic[t][4]),float(targets_dic[t][5])-30,foilhole_part_count[t][0],size=3)
    plt.savefig('{0}_targets.png'.format(GS_name),dpi=800)
    plt.close()
print(GS_imdic)

print('\nGridsquare ID\t\t#foilholes')
for  i in big_GS_dic:
    print ('{0}\t{1}'.format(i,len(big_GS_dic[i])))
print('{0} foilholes contributed particles'.format(len(big_GS_dic)))
# makeing flat lists for analysis
print('-- gathering stats for correlation analysis --')
flat_dic = {}
xs,ys,nparts,LLCs,MVPDs,defoci,means,stds,DTs = [],[],[],[],[],[],[],[],[]
thicknessdic = {}       #{GS:[[xs],[ys],[sq_means]]}
for GS in big_GS_dic:
    thicknessdic[GS] = [[],[],[]]
    print(GS)
    gridsquare_image = mrcfile.open(GS)
    micdata = gridsquare_image.data
    imdim = micdata.shape
    DTdic = {}          #{datetime:[MVPD,LLC]}
    for FH in big_GS_dic[GS]:
        sq_mean,sq_stdev = extract_square(micdata,75,big_GS_dic[GS][FH][0],big_GS_dic[GS][FH][1])
        thicknessdic[GS][2].append(sq_mean)
        flat_dic[FH] = big_GS_dic[GS][FH]
        xs.append(float(big_GS_dic[GS][FH][0]))
        thicknessdic[GS][0].append(float(big_GS_dic[GS][FH][0]))
        ys.append(float(big_GS_dic[GS][FH][1]))
        thicknessdic[GS][1].append(float(big_GS_dic[GS][FH][1]))
        nparts.append(float(big_GS_dic[GS][FH][2]))
        LLCs.append(float(big_GS_dic[GS][FH][3]))
        MVPDs.append(float(big_GS_dic[GS][FH][4]))
        defoci.append(float(big_GS_dic[GS][FH][5]))
        DTdic[big_GS_dic[GS][FH][6]] = [float(big_GS_dic[GS][FH][4]),float(big_GS_dic[GS][FH][3]),sq_mean]
        means.append(float(sq_mean))
        stds.append(float(sq_stdev))
    gridsquare_image.close()

DTdickeys = list(DTdic)
DTdickeys.sort()
DTsortedLLCs,DTsortedMVPDs,DTsortedsq_mean = [],[],[]
for i in DTdickeys:
    DTsortedLLCs.append(DTdic[i][0])
    DTsortedMVPDs.append(DTdic[i][1])
    DTsortedsq_mean.append(DTdic[i][2])
print('-- making thickness plots --')
for i in thicknessdic:
    try:
        GSname = i.split('/')[-1].split('.')[0]
        print('{0} / {1}'.format(GSname,GS_imdic[GSname]))
        make_bg(i)
        h= plt.scatter(thicknessdic[i][0],thicknessdic[i][1],c=thicknessdic[i][2],s=30,cmap='cool',vmin=min(means),vmax=max(means))
        plt.colorbar(h)
        plt.savefig('{0}_thick.png'.format(GS_imdic[GSname]))
        plt.close()
    except:
        print('skipped {0} - no foilholes'.format(GSname))
print('-- plotting --')
make_correlation_plot([xs,ys,MVPDs],['x','y','MaxValProbDist'])
make_correlation_plot([xs,ys,LLCs],['x','y','LogLikelyhoodContrib'])
make_correlation_plot([xs,ys,nparts],['x','y','numparts'])
make_correlation_plot([defoci,nparts],['defocus','numparts'])
make_correlation_plot([defoci,MVPDs],['defocus','MaxValProbDist'])
make_correlation_plot([defoci,LLCs],['defocus','LogLikelyhoodContrib'])
make_correlation_plot([MVPDs,LLCs],['MaxValueProbDist','LogLikelyhoodContrib'])

make_correlation_plot([means,LLCs],['Thickness','LogLikelyhoodContrib'])
make_correlation_plot([means,MVPDs],['Thickness','MaxValueProbDist'])
make_correlation_plot([means,nparts],['Thickness','number of particles'])
make_correlation_plot([xs,ys,means],['x','y','mean pixel value'])

make_correlation_plot([range(len(DTsortedLLCs)),DTsortedLLCs],['Aquisition order','LogLiklyhoodContrib'])
make_correlation_plot([range(len(DTsortedMVPDs)),DTsortedMVPDs],['Aquisition order','MaxValProbDist'])
make_correlation_plot([range(len(DTsortedsq_mean)),DTsortedsq_mean],['Aquisition order','Thickness'])
