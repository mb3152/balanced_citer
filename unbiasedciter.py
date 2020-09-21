"""
hi, this code is to check your citations for race biases
please contact max bertolero at mbertolero@me.com for any questions!
all you need to do is make sure you have the correct path to your bib file and where you store this repo
then type something like:  

%run unbiasedciter -authors 'Maxwell Bertolero Danielle Bassett' -bibfile '/Users/maxwell/Desktop/main.bib' -gender_key 'key from gender-api.com'

"""

import os
import pandas as pd
import tqdm
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json
import pickle
from urllib.request import urlopen
"""
some uncommon libs 
"""
try: 
	from ethnicolr import census_ln, pred_census_ln,pred_wiki_name
except:
	os.system('pip install ethnicolr')
	from ethnicolr import census_ln, pred_census_ln,pred_wiki_name
try:
	from pybtex.database import parse_file
except: 
	os.system('pip install pybtex')
	from pybtex.database import parse_file

import seaborn as sns

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-bibfile',action='store',dest='bibfile',default='example.bib')
parser.add_argument('-homedir',action='store',dest='homedir',default='//Users/maxwell/Documents/GitHub/unbiasedciter/')
parser.add_argument('-method',action='store',dest='method',default='wiki')
parser.add_argument('-authors',action='store',dest='authors')
parser.add_argument('-font',action='store',dest='font',default='Palatino') # hey, we all have our favorite
parser.add_argument('-gender_key',action='store',dest='gender_key',default=None) # hey, we all have our favorite
r = parser.parse_args()
locals().update(r.__dict__)
bibname = bibfile.split('.')[0]
bibfile = parse_file(bibfile)

wiki_2_race = {"Asian,GreaterEastAsian,EastAsian":'api', "Asian,GreaterEastAsian,Japanese":'api',
"Asian,IndianSubContinent":'api', "GreaterAfrican,Africans":'black', "GreaterAfrican,Muslim":'black',
"GreaterEuropean,British":'white', "GreaterEuropean,EastEuropean":'white',
"GreaterEuropean,Jewish":'white', "GreaterEuropean,WestEuropean,French":'white',
"GreaterEuropean,WestEuropean,Germanic":'white', "GreaterEuropean,WestEuropean,Hispanic":'hispanic',
"GreaterEuropean,WestEuropean,Italian":'white', "GreaterEuropean,WestEuropean,Nordic":'white'}

def gender_base():
	"""
	for unknown gender, fill with base rates
	you will never / can't run this (that file is too big to share)
	"""
	main_df = pd.read_csv('/%s/data/NewArticleData2019.csv'%(homedir),header=0)


	gender_base = {}
	for year in np.unique(main_df.PY.values):
		ydf = main_df[main_df.PY==year].AG
		fa = np.array([x[0] for x in ydf.values])
		la = np.array([x[1] for x in ydf.values])

		fa_m = len(fa[fa=='M'])/ len(fa[fa!='U'])
		fa_w = len(fa[fa=='W'])/ len(fa[fa!='U'])

		la_m = len(la[fa=='M'])/ len(la[la!='U'])
		la_w = len(la[fa=='W'])/ len(la[la!='U'])

		gender_base[year] = [fa_m,fa_w,la_m,la_w]

	gender_base[2020] = [fa_m,fa_w,la_m,la_w]

	with open(homedir + '/data/gender_base' + '.pkl', 'wb') as f:
		pickle.dump(gender_base, f, pickle.HIGHEST_PROTOCOL)


with open(homedir + 'data/gender_base' + '.pkl', 'rb') as f:
	gender_base =  pickle.load(f)

authors = authors.split(' ')
print ('first author is %s %s '%(authors[0],authors[1]))
print ('last author is %s %s '%(authors[2],authors[3]))
print ("we don't count these")

citation_matrix = np.zeros((8,8))
matrix_idxs = {'white_m':0,'api_m':1,'hispanic_m':2,'black_m':3,'white_f':4,'api_f':5,'hispanic_f':6,'black_f':7}

asian = [0,1,2]
black = [3,4]
white = [5,6,7,8,9,11,12]
hispanic = [10]
print ('looping through your references, predicting gender and race')

columns=['Reference Key','Author','Gender','W','A']
paper_df = pd.DataFrame(columns=columns)

gender = []
race = []


idx = 0
for paper in tqdm.tqdm(bibfile.entries,total=len(bibfile.entries)): 
	if 'author' not in bibfile.entries[paper].persons.keys():
		continue #some editorials have no authors
	if 'year' not in bibfile.entries[paper].fields.keys():
		year = 2020
	else: year = int(bibfile.entries[paper].fields['year'])  
	
	if year not in gender_base.keys():
		gb = gender_base[1995]
	else:
		gb = gender_base[year]

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

	fa_fname = fa_fname.encode("ascii", errors="ignore").decode() 
	fa_lname = fa_lname.encode("ascii", errors="ignore").decode()
	la_fname = la_fname.encode("ascii", errors="ignore").decode() 
	la_lname = la_lname.encode("ascii", errors="ignore").decode() 

	names = [{'lname': fa_lname,'fname':fa_fname}]
	fa_df = pd.DataFrame(names,columns=['fname','lname'])
	fa_race = pred_wiki_name(fa_df,'fname','lname').values[0][3:]
	fa_race = [np.sum(fa_race[white]),np.sum(fa_race[asian]),np.sum(fa_race[hispanic]),np.sum(fa_race[black])]
	
	names = [{'lname': la_lname,'fname':la_fname}]
	la_df = pd.DataFrame(names,columns=['fname','lname'])
	la_race = pred_wiki_name(la_df,'fname','lname').values[0][3:]
	la_race = [np.sum(la_race[white]),np.sum(la_race[asian]),np.sum(la_race[hispanic]),np.sum(la_race[black])]

	url = "https://gender-api.com/get?key=" + gender_key + "&name=%s" %(fa_fname)
	response = urlopen(url)
	decoded = response.read().decode('utf-8')
	fa_gender = json.loads(decoded)
	if fa_gender['gender'] == 'female':
		fa_g = [0,fa_gender['accuracy']/100.]
	if fa_gender['gender'] == 'male':
		fa_g = [fa_gender['accuracy']/100.,0]
	if fa_gender['gender'] == 'unknown':
		fa_g = gb[:2]

	url = "https://gender-api.com/get?key=" + gender_key + "&name=%s" %(la_fname)
	response = urlopen(url)
	decoded = response.read().decode('utf-8')
	la_gender = json.loads(decoded)
	if la_gender['gender'] == 'female':
		la_g = [0,la_gender['accuracy']/100.]
	
	if la_gender['gender'] == 'male':
		la_g = [la_gender['accuracy']/100.,0]

	if la_gender['gender'] == 'unknown':
		la_g = gb[2:] 
	
	fa_data = np.array([paper,'%s,%s'%(fa_fname,fa_lname),'%s,%s'%(fa_gender['gender'],fa_gender['accuracy']),fa_race[0],np.sum(fa_race[1:])]).reshape(1,5)
	paper_df = paper_df.append(pd.DataFrame(fa_data,columns=columns),ignore_index =True)
	la_data = np.array([paper,'%s,%s'%(la_fname,la_lname),'%s,%s'%(la_gender['gender'],la_gender['accuracy']),la_race[0],np.sum(la_race[1:])]).reshape(1,5)
	paper_df = paper_df.append(pd.DataFrame(la_data,columns=columns),ignore_index =True)

	mm = fa_g[0]*la_g[0]
	wm = fa_g[1]*la_g[0]
	mw = fa_g[0]*la_g[1]
	ww = fa_g[1]*la_g[1]
	mm,wm,mw,ww = [mm,wm,mw,ww]/np.sum([mm,wm,mw,ww])
	
	gender.append([mm,wm,mw,ww])

	ww = fa_race[0] * la_race[0]
	aw = np.sum(fa_race[1:]) * la_race[0]
	wa = fa_race[0] * np.sum(la_race[1:])
	aa = np.sum(fa_race[1:]) * np.sum(la_race[1:])

	race.append([ww,aw,wa,aa])

	paper_matrix = np.zeros((2,8))
	paper_matrix[0] = np.outer(fa_g,fa_race).flatten() 
	paper_matrix[1] = np.outer(la_g,la_race).flatten() 

	paper_matrix = np.outer(paper_matrix[0],paper_matrix[1]) 

	citation_matrix = citation_matrix + paper_matrix
	idx = idx + 1

mm,wm,mw,ww = np.mean(gender,axis=0)*100
WW,aw,wa,aa = np.mean(race,axis=0)*100

statement = "Recent work in several fields of science has identified a bias in citation practices such that papers from women and other minority scholars\
are under-cited relative to the number of such papers in the field (1-5). Here we sought to proactively consider choosing references that reflect the \
diversity of the field in thought, form of contribution, gender, race, ethnicity, and other factors. First, we obtained the predicted gender of the first \
and last author of each reference by using databases that store the probability of a first name being carried by a woman (5, 6). By this measure \
(and excluding self-citations to the first and last authors of our current paper), our references contain ww% woman(first)/woman(last), \
MW% man/woman, WM% woman/man, and MM% man/man. This method is limited in that a) names, pronouns, and social media profiles used to construct the \
databases may not, in every case, be indicative of gender identity and b) it cannot account for intersex, non-binary, or transgender people. \
Second, we obtained predicted racial/ethnic category of the first and last author of each reference by databases that store the probability of a \
first and last name being carried by an author of color (7,8). By this measure (and excluding self-citations), our references contain AA% author of \
color (first)/author of color(last), WA% white author/author of color, AW% author of color/white author, and WW% white author/white author. This method \
is limited in that a) names and Wikipedia profiles used to make the predictions may not be indicative of racial/ethnic identity, and b) \
it cannot account for Indigenous and mixed-race authors, or those who may face differential biases due to the ambiguous racialization or ethnicization of their names.  \
We look forward to future work that could help us to better understand how to support equitable practices in science."

statement = statement.replace('MM',str(np.around(mm,2)))
statement = statement.replace('WM',str(np.around(wm,2)))
statement = statement.replace('MW',str(np.around(mw,2)))
statement = statement.replace('ww',str(np.around(ww,2)))
statement = statement.replace('WW',str(np.around(WW,2)))
statement = statement.replace('AW',str(np.around(aw,2)))
statement = statement.replace('WA',str(np.around(wa,2)))
statement = statement.replace('AA',str(np.around(aa,2)))

print (statement)


cmap = sns.diverging_palette(220, 10, as_cmap=True)
names = ['white_m','api_m','hispanic_m','black_m','white_w','api_w','hispanic_w','black_w']
plt.close()
sns.set(style='white',font=font)
fig, axes = plt.subplots(ncols=2,nrows=1,figsize=(7.5,4))
axes = axes.flatten()
plt.sca(axes[0])
heat = sns.heatmap(np.around((citation_matrix/citation_matrix.sum())*100,2),annot=True,ax=axes[0],annot_kws={"size": 8},cmap=cmap,vmax=1,vmin=0)
axes[0].set_ylabel('first author',labelpad=0)  
heat.set_yticklabels(names,rotation=0)
axes[0].set_xlabel('last author',labelpad=1)  
heat.set_xticklabels(names,rotation=90) 
heat.set_title('percentage of citations')  

citation_matrix_sum = citation_matrix / np.sum(citation_matrix) 

expected = np.load('/%s/data/expected_matrix_wiki.npy'%(homedir))
expected = expected/np.sum(expected)

percent_overunder = np.ceil( ((citation_matrix_sum - expected) / expected)*100)
plt.sca(axes[1])
heat = sns.heatmap(np.around(percent_overunder,2),annot=True,ax=axes[1],fmt='g',annot_kws={"size": 8},vmax=50,vmin=-50,cmap=cmap)
axes[1].set_ylabel('',labelpad=0)  
heat.set_yticklabels([''])
axes[1].set_xlabel('last author',labelpad=1)  
heat.set_xticklabels(names,rotation=90) 
heat.set_title('percentage over/under-citations')
plt.tight_layout()

plt.savefig('/%s/data/race_gender_citations_%s.pdf'%(homedir,bibname))





