"""
hi, this code is to check your citations for race biases
please contact max bertolero at mbertolero@me.com for any questions!

all you need to do is make sure you have the correct path to your bib file and where you store this repo
then type something like:  %run unbiasedciter -authors 'Maxwell Bertolero Danielle Bassett' -bibfile '/Users/maxwell/Desktop/main.bib' 

"""
import os
import pandas as pd
import tqdm
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import numpy as np
"""
some uncommon libs 
"""
from ethnicolr import census_ln, pred_census_ln,pred_wiki_name
from pybtex.database import parse_file
import seaborn as sns

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-bibfile',action='store',dest='bibfile',default='main.bib')
parser.add_argument('-homedir',action='store',dest='homedir',default='//Users/maxwell/Dropbox/Bertolero_Bassett_Projects/unbiasedciter/')
parser.add_argument('-method',action='store',dest='method',default='wiki')
parser.add_argument('-authors',action='store',dest='authors')
parser.add_argument('-font',action='store',dest='font',default='Palatino') # hey, we all have our favorite
r = parser.parse_args()
locals().update(r.__dict__)
bibfile = parse_file(bibfile)

wiki_2_race = {"Asian,GreaterEastAsian,EastAsian":'api', "Asian,GreaterEastAsian,Japanese":'api',
"Asian,IndianSubContinent":'api', "GreaterAfrican,Africans":'black', "GreaterAfrican,Muslim":'black',
"GreaterEuropean,British":'white', "GreaterEuropean,EastEuropean":'white',
"GreaterEuropean,Jewish":'white', "GreaterEuropean,WestEuropean,French":'white',
"GreaterEuropean,WestEuropean,Germanic":'white', "GreaterEuropean,WestEuropean,Hispanic":'hispanic',
"GreaterEuropean,WestEuropean,Italian":'white', "GreaterEuropean,WestEuropean,Nordic":'white'}


citation_matrix = np.zeros((4,4))
small_matrix = np.zeros((2,2))
matrix_idxs = {'white':0,'api':1,'hispanic':2,'black':3}
small_idxs = {'white':0,'api':1,'hispanic':1,'black':1}

authors = authors.split(' ')

print ('first author is %s %s '%(authors[0],authors[1]))
print ('last author is %s %s '%(authors[2],authors[2]))
print ("we don't count these")


for paper in tqdm.tqdm(bibfile.entries,total=len(bibfile.entries)): 
	fa = bibfile.entries[paper].persons['author'][0]
	try:fa_fname = fa.first_names[0] 
	except:fa_fname = fa.last_names[0] #for people like Plato
	fa_lname = fa.last_names[0] 
	la = bibfile.entries[paper].persons['author'][-1]
	try:la_fname = la.first_names[0] 
	except:la_fname = la.last_names[0] #for people like Plato
	la_lname = la.last_names[0]

	if fa_fname == authors[0]:
		if fa_lname == authors[1]:
			continue

	if fa_fname == authors[2]:
		if fa_lname == authors[3]:
			continue

	if la_fname == authors[0]:
		if la_lname == authors[1]:
			continue
	
	if la_fname == authors[2]:
		if la_lname == authors[3]:
			continue

	##wiki
	if method =='wiki':
		names = [{'lname': fa_lname,'fname':fa_fname}]
		fa_df = pd.DataFrame(names,columns=['fname','lname'])
		fa_race = wiki_2_race[pred_wiki_name(fa_df,'fname','lname').race.values[0]]
		names = [{'lname': la_lname,'fname':la_fname}]
		la_df = pd.DataFrame(names,columns=['fname','lname'])
		la_race = wiki_2_race[pred_wiki_name(la_df,'fname','lname').race.values[0]]
	#census
	if method =='census':
		names = [{'name': fa_lname},{'name':la_lname}]
		la_df = pd.DataFrame(names)
		r = pred_census_ln(la_df,'name')
		fa_race,la_race= r.race.values
	citation_matrix[matrix_idxs[fa_race],matrix_idxs[la_race]] +=1
	small_matrix[small_idxs[fa_race],small_idxs[la_race]] +=1
# 1/0
plt.close()
sns.set(style='white',font=font)
fig, axes = plt.subplots(ncols=2,nrows=2,figsize=(7.5,6))
heat = sns.heatmap(citation_matrix,annot=True,ax=axes[0,0])
axes[0,0].set_ylabel('first author',labelpad=0)  
heat.set_yticklabels(['white','api','hispanic','black'],rotation=65)
axes[0,0].set_xlabel('last author',labelpad=0)  
heat.set_xticklabels(['white','api','hispanic','black']) 
heat.set_title('# of citations')  

citation_matrix = citation_matrix / np.sum(citation_matrix) 

expected = np.load('/%s/data/expected_matrix_%s.npy'%(homedir,method))
expected = expected/np.sum(expected)

percent_overunder = np.ceil( ((citation_matrix - expected) / expected)*100)

heat = sns.heatmap((percent_overunder).astype(int),annot=True,ax=axes[0,1],fmt='g')
axes[0,1].set_ylabel('first author',labelpad=0)  
heat.set_yticklabels(['white','api','hispanic','black'],rotation=65)
axes[0,1].set_xlabel('last author',labelpad=0)  
heat.set_xticklabels(['white','api','hispanic','black']) 
heat.set_title('percentage over/under-citations')

# 1/0
heat = sns.heatmap(small_matrix,annot=True,ax=axes[1,0])
axes[0,0].set_ylabel('first author',labelpad=0)  
heat.set_yticklabels(['white','poc'])
axes[0,0].set_xlabel('last author',labelpad=0)  
heat.set_xticklabels(['white','poc'])
heat.set_title('# of citations')  

small_matrix = small_matrix / np.sum(small_matrix) 

expected = np.load('/%s/data/expected_small_matrix_%s.npy'%(homedir,method))
expected = expected/np.sum(expected)

percent_overunder = np.ceil( ((small_matrix - expected) / expected)*100)
heat.set_title('percentage over/under-citations')
heat = sns.heatmap((percent_overunder).astype(int),annot=True,ax=axes[1,1],fmt='g')
axes[0,1].set_ylabel('first author',labelpad=0)  
heat.set_yticklabels(['white','poc'])
axes[0,1].set_xlabel('last author',labelpad=0)  
heat.set_xticklabels(['white','poc'])



plt.tight_layout()
# 1/0
plt.savefig('race_citations.pdf')
