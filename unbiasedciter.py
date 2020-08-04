import pandas as pd
import os
import time
try:from ethnicolr import census_ln, pred_census_ln,pred_wiki_name
except: os.system('pip install ethnicolr')
import seaborn as sns
import matplotlib.pylab as plt
import scipy
from itertools import permutations      
import numpy as np
import matplotlib.gridspec as gridspec
from igraph import VertexClustering
from itertools import combinations 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = "Palatino"
plt.rcParams['font.serif'] = "Palatino"
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Palatino:italic'
plt.rcParams['mathtext.bf'] = 'Palatino:bold'
plt.rcParams['mathtext.cal'] = 'Palatino'
from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor 
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.linear_model import RidgeClassifierCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import RidgeCV

from statsmodels.stats.multitest import multipletests

import multiprocessing
from multiprocessing import Pool 
import tqdm
import igraph
from scipy.stats import pearsonr 


global paper_df
global main_df
global g
global graphs
global pal
global homedir
global method
global node_2_a
global a_2_node
global a_2_paper
global control
global matrix_idxs
global prs
# matrix_idxs = {'white_M':0,'white_W':1,'white_U':2,'api_M':3,'api_W':4,'api_U':5,'hispanic_M':6,'hispanic_W':7,'hispanic_U':8,'black_M':9,'black_W':10,'black_U':11}


pal = np.array([[72,61,139],[82,139,139],[180,205,205],[205,129,98]])/255.

# global us_only
# us_only = True

"""
AF = author names, with the format LastName, FirstName; LastName, FirstName; etc..
SO = journal
DT = document type (review or article)
CR = reference list
TC = total citations received (at time of downloading about a year ago)
PD = month of publication
PY = year of publication
DI = DOI
"""

import argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return v

parser = argparse.ArgumentParser()
parser.add_argument('-homedir',action='store',dest='homedir',default='/Users/maxwell/Dropbox/Bertolero_Bassett_Projects/citations/')
parser.add_argument('-method',action='store',dest='method',default='wiki')
parser.add_argument('-continent',type=str2bool,action='store',dest='continent',default=False)
parser.add_argument('-continent_only',type=str2bool,action='store',dest='continent_only',default=False)
parser.add_argument('-control',type=str2bool,action='store',dest='control',default=False)
parser.add_argument('-within_poc',type=str2bool,action='store',dest='within_poc',default=False)
parser.add_argument('-walk_length',type=str,action='store',dest='walk_length',default='cited')
parser.add_argument('-walk_papers',type=str2bool,action='store',dest='walk_papers',default=False)

r = parser.parse_args()
locals().update(r.__dict__)
globals().update(r.__dict__)

wiki_2_race = {"Asian,GreaterEastAsian,EastAsian":'api', "Asian,GreaterEastAsian,Japanese":'api',
"Asian,IndianSubContinent":'api', "GreaterAfrican,Africans":'black', "GreaterAfrican,Muslim":'black',
"GreaterEuropean,British":'white', "GreaterEuropean,EastEuropean":'white',
"GreaterEuropean,Jewish":'white', "GreaterEuropean,WestEuropean,French":'white',
"GreaterEuropean,WestEuropean,Germanic":'white', "GreaterEuropean,WestEuropean,Hispanic":'hispanic',
"GreaterEuropean,WestEuropean,Italian":'white', "GreaterEuropean,WestEuropean,Nordic":'white'}
matrix_idxs = {'white_M':0,'api_M':1,'hispanic_M':2,'black_M':3,'white_W':4,'api_W':5,'hispanic_W':6,'black_W':7}

def log_p_value(p):
	if p == 0.0:
		p = "-log10($\it{p}$)>250"
	elif p > 0.001: 
		p = np.around(p,3)
		p = "$\it{p}$=%s"%(p)
	else: 
		p = (-1) * np.log10(p)
		p = "-log10($\it{p}$)=%s"%(np.around(p,0).astype(int))
	return p

def convert_r_p(r,p):
	return "$\it{r}$=%s\n%s"%(np.around(r,2),log_p_value(p))

def nan_pearsonr(x,y):
	xmask = np.isnan(x)
	ymask = np.isnan(y)
	mask = (xmask==False) & (ymask==False) 
	return pearsonr(x[mask],y[mask])

def make_df(method=method):

	"""
	this makes the actual data by pulling the race from the census or wiki data
	"""
	if os.path.exists('/%s/data/result_df_%s.csv'%(homedir,method)):
		df = pd.read_csv('/%s/data/result_df_%s.csv'%(homedir,method))
		return df
	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	result_df = pd.DataFrame(columns=['fa_race','la_race','citation_count'])
	store_fa_race = []
	store_la_race = []
	store_citations = []
	store_year = []
	store_journal = []
	store_fa_g = []
	store_la_g = []
	store_fa_category = []
	store_la_category = []
	for entry in tqdm.tqdm(main_df.iterrows(),total=len(main_df)):
		try:ncites= len(entry[1].CP.split(',')) 
		except:ncites= 0
		store_year.append(entry[1]['PY'])
		store_journal.append(entry[1]['SO'])
		fa = entry[1].AF.split(';')[0]
		la = entry[1].AF.split(';')[-1]
		fa_lname,fa_fname = fa.split(', ')
		la_lname,la_fname = la.split(', ')
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
		store_citations.append(ncites)
		store_la_race.append(la_race)
		store_fa_race.append(fa_race)
		store_fa_g.append(entry[1].AG[0])
		store_la_g.append(entry[1].AG[1])
		store_fa_category.append('%s_%s' %(fa_race,entry[1].AG[0]))
		store_la_category.append('%s_%s' %(la_race,entry[1].AG[1]))
	result_df['fa_race'] = store_fa_race 
	result_df['la_race'] = store_la_race
	result_df['fa_g'] = store_fa_g
	result_df['la_g'] = store_la_g
	result_df['citation_count'] = store_citations
	result_df['journal'] = store_journal
	result_df['year'] = store_year
	result_df['fa_category'] = store_fa_category
	result_df['la_category'] = store_la_category
	result_df.citation_count = result_df.citation_count.values.astype(int) 
	result_df.to_csv('/%s/data/result_df_%s.csv'%(homedir,method),index=False)
	return result_df

def make_pr_df(method=method):

	"""
	this makes the actual data by pulling the race from the census or wiki data
	"""
	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	prs = np.zeros((main_df.shape[0],8,8))

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

	asian = [0,1,2]
	black = [3,4]
	white = [5,6,7,8,9,11,12]
	hispanic = [10]
	for entry in tqdm.tqdm(main_df.iterrows(),total=len(main_df)):
	
		try:ncites= len(entry[1].CP.split(';')) 
		except:ncites= 0
		fa = entry[1].AF.split(';')[0]
		la = entry[1].AF.split(';')[-1]
		fa_lname,fa_fname = fa.split(', ')
		la_lname,la_fname = la.split(', ')
		fa_g = entry[1].AG[0]   
		la_g = entry[1].AG[1]	
		paper_matrix = np.zeros((2,8))
		##wiki
		if method =='wiki':
			names = [{'lname': fa_lname,'fname':fa_fname}]
			fa_df = pd.DataFrame(names,columns=['fname','lname'])
			fa_race = pred_wiki_name(fa_df,'fname','lname').values[0][3:]
			fa_race = [np.sum(fa_race[white]),np.sum(fa_race[asian]),np.sum(fa_race[hispanic]),np.sum(fa_race[black])]
			names = [{'lname': la_lname,'fname':la_fname}]
			la_df = pd.DataFrame(names,columns=['fname','lname'])
			la_race = pred_wiki_name(la_df,'fname','lname').values[0][3:]
			la_race = [np.sum(la_race[white]),np.sum(la_race[asian]),np.sum(la_race[hispanic]),np.sum(la_race[black])]


		# #census
		if method =='census':
			names = [{'name': fa_lname},{'name':la_lname}]
			la_df = pd.DataFrame(names)
			r = pred_census_ln(la_df,'name')
			fa_race = [r.iloc[0]['white'],r.iloc[0]['api'],r.iloc[0]['hispanic'],r.iloc[0]['black']]
			la_race = [r.iloc[1]['white'],r.iloc[1]['api'],r.iloc[1]['hispanic'],r.iloc[1]['black']]
		gender_b = gender_base[year]
		if fa_g == 'M': paper_matrix[0] = np.outer([1,0],fa_race).flatten() 
		if fa_g == 'W': paper_matrix[0] = np.outer([0,1],fa_race).flatten() 
		if fa_g == 'U': paper_matrix[0] = np.outer([gender_b[0],gender_b[1]],fa_race).flatten() 

		if la_g == 'M': paper_matrix[1] = np.outer([1,0],la_race).flatten() 
		if la_g == 'W': paper_matrix[1] = np.outer([0,1],la_race).flatten() 
		if la_g == 'U': paper_matrix[1] = np.outer([gender_b[2],gender_b[3]],la_race).flatten() 

		paper_matrix = np.outer(paper_matrix[0],paper_matrix[1]) 
		paper_matrix = paper_matrix / np.sum(paper_matrix)
		prs[entry[0]] = paper_matrix
	
	np.save('/%s/data/result_pr_df_%s.npy'%(homedir,method),prs)

def make_all_author_race():

	"""
	this makes the actual data by pulling the race from the census or wiki data,
	but this version include middle authors, which we use for the co-authorship networks
	"""
	# if os.path.exists('/%s/data/result_df_%s_all.csv'%(homedir,method)):
	# 	df = pd.read_csv('/%s/data/result_df_%s_all.csv'%(homedir,method))
	# 	return df
	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	names = []
	lnames = []
	fnames = []
	for entry in main_df.iterrows():
		for a in entry[1].AF.split('; '):
			##wiki
			if method =='wiki':
				a_lname,a_fname = a.split(', ')
				lnames.append(a_lname.strip())
				fnames.append(a_fname.strip())
				names.append(a)
			#census
			if method =='census':
				a_lname,a_fname = a.split(', ')
				names.append(a_lname.strip())



	if method =='census':
		df = pd.DataFrame(columns=['name'])
		df.name = np.unique(names)		
		r = pred_census_ln(df,'name')
		r.to_csv('/%s/data/result_df_%s_all.csv'%(homedir,method),index=False)
	
		all_races = []
		r = dict(zip(df.name.values,df.race.values)) 
		for idx,paper in tqdm.tqdm(main_df.iterrows(),total=main_df.shape[0]):
			races = []
			for a in paper.AF.split('; '):
				a_lname,a_fname = a.split(', ')
				races.append(r[a_lname.strip()])
			all_races.append('_'.join(str(x) for x in races))
		main_df['all_races'] = all_races 
		main_df.to_csv('/%s/data/all_data_%s.csv'%(homedir,method),index=False)


	if method =='wiki':
		all_races = []
		df = pd.DataFrame(np.array([names,fnames,lnames]).swapaxes(0,1),columns=['name','fname','lname'])
		df = df.drop_duplicates('name') 
		r = pred_wiki_name(df,'fname','lname')
		r.race = r.race.map(wiki_2_race) 
		r.to_csv('/%s/data/result_df_%s_all.csv'%(homedir,method),index=False)
		for idx,paper in tqdm.tqdm(main_df.iterrows(),total=main_df.shape[0]):
			races = []
			for a in paper.AF.split('; '):
				races.append(r[r.name==a].race.values[0])
			all_races.append('_'.join(str(x) for x in races))
		main_df['all_races'] = all_races
		main_df.to_csv('/%s/data/all_data_%s.csv'%(homedir,method),index=False)

def figure_1():
	n_iters = 1000
	df = make_df()
	prs = np.load('/%s/data/result_pr_df_%s.npy'%(homedir,method))
	results = np.zeros((len(np.unique(df.year)),4))
	for yidx,year in enumerate(np.unique(df.year)):
		data = df[df.year==year]
		ww = len(data[(data.fa_race=='white')&(data.la_race=='white')].citation_count.values)
		wn = len(data[(data.fa_race=='white')&(data.la_race!='white')].citation_count.values)
		nw = len(data[(data.fa_race!='white')&(data.la_race=='white')].citation_count.values)
		nn = len(data[(data.fa_race!='white')&(data.la_race!='white')].citation_count.values)
		total = float(ww + wn + nw + nn)
		results[yidx,0] = ww / total
		results[yidx,1] = wn / total
		results[yidx,2] = nw / total
		results[yidx,3] = nn / total
	
	plt.close()
	sns.set(style='white',font='Palatino')
	# pal = sns.color_palette("Set2")
	# pal = sns.color_palette("vlag",4)
	fig = plt.figure(figsize=(7.5,4),constrained_layout=False)
	gs = gridspec.GridSpec(15, 15, figure=fig,wspace=.1,hspace=0,left=.1,right=.9,top=.9,bottom=.1)
	labels = ['white author & white author','white author & author of color',\
	'author of color & white author','author of color & author of color']
	ax1 = fig.add_subplot(gs[:15,:5])
	plt.sca(ax1)
	ax1_plot = plt.stackplot(np.unique(df.year),results.transpose()*100, labels=labels,colors=pal, alpha=1)
	handles, labels = plt.gca().get_legend_handles_labels()
	labels.reverse()
	handles.reverse()
	leg = plt.legend(loc=8,frameon=False,labels=labels,handles=handles,fontsize=8)
	for text in leg.get_texts():
		plt.setp(text, color = 'w')
	plt.margins(0,0)
	plt.ylabel('percentage of citations')
	plt.xlabel('publication year')
	ax1.tick_params(axis='x', which='major', pad=-5)
	ax1.tick_params(axis='y', which='major', pad=0)
	i,j,k,l = results[0]*100
	i,j,k,l = [i,(i+j),(i+j+k),(i+j+k+l)]
	i,j,k,l = [np.mean([0,i]),np.mean([i,j]),np.mean([j,k]),np.mean([k,l])]
	# i,j,k,l = np.array([100]) - np.array([i,j,k,l])  
	plt.sca(ax1)	
	ax1.yaxis.set_major_formatter(ticker.PercentFormatter()) 
	ax1.set_yticks([i,j,k,l])
	ax1.set_yticklabels(np.around(results[0]*100,0).astype(int))

	ax2 = ax1_plot[0].axes.twinx()
	plt.sca(ax2)
	i,j,k,l = results[-1]*100
	i,j,k,l = [i,(i+j),(i+j+k),(i+j+k+l)]
	i,j,k,l = [np.mean([0,i]),np.mean([i,j]),np.mean([j,k]),np.mean([k,l])] 
	plt.ylim(0,100)
	ax2.yaxis.set_major_formatter(ticker.PercentFormatter()) 
	ax2.set_yticks([i,j,k,l])
	ax2.set_yticklabels(np.around(results[-1]*100,0).astype(int))
	plt.xticks([1995., 2000., 2005., 2010., 2015., 2019],np.array([1995., 2000., 2005., 2010., 2015., 2019]).astype(int))   

	ax2.tick_params(axis='y', which='major', pad=0)
	plt.title('a',{'fontweight':'bold'},'left',pad=1)

	axes = []
	jidx = 3
	for makea in range(5):
		axes.append(fig.add_subplot(gs[jidx-3:jidx,6:10]))
		jidx=jidx+3
	
	for aidx,journal in enumerate(np.unique(df.journal)):
		results = np.zeros(( len(np.unique(df[(df.journal==journal)].year)),4))
		for yidx,year in enumerate(np.unique(df[(df.journal==journal)].year)):
			data = df[(df.year==year)&(df.journal==journal)]
			ww = len(data[(data.fa_race=='white')&(data.la_race=='white')].citation_count.values)
			wn = len(data[(data.fa_race=='white')&(data.la_race!='white')].citation_count.values)
			nw = len(data[(data.fa_race!='white')&(data.la_race=='white')].citation_count.values)
			nn = len(data[(data.fa_race!='white')&(data.la_race!='white')].citation_count.values)
			total = float(ww + wn + nw + nn)
			if total == 0.0:continue
			results[yidx,0] = ww / total
			results[yidx,1] = wn / total
			results[yidx,2] = nw / total
			results[yidx,3] = nn / total
		data = df[df.journal==journal]
		ax = axes[aidx]
		plt.sca(ax)
		ax1_plot = plt.stackplot(np.unique(data.year),results.transpose()*100, labels=labels,colors=pal, alpha=1)
		plt.margins(0,0)
		ax.set_yticks([])
		ax.set_xticks([])
		plt.title(journal.title(), pad=-40,color='w',fontsize=8)
		if aidx == 0: plt.text(0,1,'b',{'fontweight':'bold'},horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes)

	df = make_df()
	journals = np.unique(df.journal)
	plot_df = pd.DataFrame(columns=['journal','year','percentage','iteration'])  
	for j in journals:
		for yidx,year in enumerate(np.unique(df.year)):
			for i in range(n_iters):
				data = df[(df.year==year)&(df.journal==j)]
				data = data.sample(int(len(data)),replace=True)
				ww = len(data[(data.fa_race=='white')&(data.la_race=='white')].citation_count.values)
				wn = len(data[(data.fa_race=='white')&(data.la_race!='white')].citation_count.values)
				nw = len(data[(data.fa_race!='white')&(data.la_race=='white')].citation_count.values)
				nn = len(data[(data.fa_race!='white')&(data.la_race!='white')].citation_count.values)
				total = float(ww + wn + nw + nn)
				r = np.array([wn + nw + nn])/total
				r = r.sum()
				tdf = pd.DataFrame(np.array([j,r,year,i]).reshape(1,-1),columns=['journal','percentage','year','iteration']) 
				plot_df = plot_df.append(tdf,ignore_index=True)


	plot_df.percentage = plot_df.percentage.astype(float)
	plot_df.iteration = plot_df.iteration.astype(int)
	plot_df.percentage = plot_df.percentage.astype(float) * 100
	pct_df = pd.DataFrame(columns=['journal','year','percentage','iteration'])
	plot_df = plot_df.sort_values('year')
	for i in range(n_iters):
		for j in journals:
			change = np.diff(plot_df[(plot_df.iteration==i)&(plot_df.journal==j)].percentage)
			tdf = pd.DataFrame(columns=['journal','year','percentage','iteration'])
			tdf.year = range(1997,2020)
			tdf.percentage = change[1:]
			tdf.journal = j
			tdf.iteration = i
			pct_df = pct_df.append(tdf,ignore_index=True)


	pct_df = pct_df.dropna()
	pct_df = pct_df[np.isinf(pct_df.percentage)==False] 
	mean_pc = pct_df.groupby(['journal']).mean() 
	min_pc = mean_pc - pct_df.groupby(['journal']).std() *2
	max_pc = mean_pc + pct_df.groupby(['journal']).std() *2  
	mean_pc = np.around(mean_pc.values,0)
	min_pc = np.around(min_pc.values,0)
	max_pc = np.around(max_pc.values,0)

	axes = []
	jidx = 3
	for makea in range(5):
		axes.append(fig.add_subplot(gs[jidx-3:jidx,10:]))
		jidx=jidx+3 

	for i,ax,journal,color in zip(range(5),axes,journals,sns.color_palette("rocket_r", 5)):
		plt.sca(ax)
		if i == 0: plt.title('c',{'fontweight':'bold'},'left',pad=1)
		lp = sns.lineplot(data=plot_df[plot_df.journal==journal],y='percentage',x='year',color=color,ci=100)   
		plt.margins(0,0)
		ax.set_yticks([])
		ax.set_xticks([])
		plt.ylabel('')
		plt.xlabel('')
		thisdf = plot_df[plot_df.journal==journal]
		minp = int(np.around(thisdf.mean()['percentage'],0))
		thisdf = thisdf[thisdf.year==thisdf.year.max()]
		maxp = int(np.around(thisdf.mean()['percentage'],0))
		plt.text(0.01,.75,'%s'%(minp),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes,fontsize=8)
		plt.text(1,.9,'%s'%(maxp),horizontalalignment='right',verticalalignment='top', transform=ax.transAxes,fontsize=8)
		plt.text(.99,0,'95%' + "CI: %s<%s>%s"%(min_pc[i][0],mean_pc[i][0],max_pc[i][0]),horizontalalignment='right',verticalalignment='bottom', transform=ax.transAxes,fontsize=8)
	plt.savefig('/%s/figures/figure1_%s.pdf'%(homedir,method))

def figure_1_pr():
	n_iters = 1000
	df = make_df()
	matrix = np.load('/%s/data/result_pr_df_%s.npy'%(homedir,method))
	results = np.zeros((len(np.unique(df.year)),4))

	if within_poc == False:
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W',]),
		np.vectorize(matrix_idxs.get)(['api_M','api_W','hispanic_M','hispanic_W','black_M','black_W',])]
		names = ['white-white','white-poc','poc-white','poc-poc']

	if within_poc == 'black':
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W','api_M','api_W','hispanic_M','hispanic_W',]),
		np.vectorize(matrix_idxs.get)(['black_M','black_W',])]
		names = ['nb-nb','nb-black','black-nb','black-black']

	if within_poc == 'api':
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W','hispanic_M','hispanic_W','black_M','black_W',]),
		np.vectorize(matrix_idxs.get)(['api_M','api_W',])]
		names = ['na-na','na-asian','asian-na','asian-asian']

	if within_poc == 'hispanic':
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W','api_M','api_W','black_M','black_W',]),
		np.vectorize(matrix_idxs.get)(['hispanic_M','hispanic_W',])]
		names = ['nh-nh','nh-hispanic','hispanic-nh','hispanic-hispanic']


	plot_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))
	plot_base_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))

	for i in range(len(groups)):
		for j in range(len(groups)):
			plot_matrix[:,i,j] = np.nansum(matrix[:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)

	for yidx,year in enumerate(np.unique(df.year)):
		papers = df[df.year==year].index  
		r = np.mean(plot_matrix[papers],axis=0).flatten()
		results[yidx,0] = r[0]
		results[yidx,1] = r[1]
		results[yidx,2] = r[2]
		results[yidx,3] = r[3]
	
	plt.close()
	sns.set(style='white',font='Palatino')
	# pal = sns.color_palette("Set2")
	# pal = sns.color_palette("vlag",4)
	fig = plt.figure(figsize=(7.5,4),constrained_layout=False)
	gs = gridspec.GridSpec(15, 15, figure=fig,wspace=.1,hspace=0,left=.1,right=.9,top=.9,bottom=.1)
	labels = ['white author & white author','white author & author of color',\
	'author of color & white author','author of color & author of color']
	ax1 = fig.add_subplot(gs[:15,:5])
	plt.sca(ax1)
	ax1_plot = plt.stackplot(np.unique(df.year),results.transpose()*100, labels=labels,colors=pal, alpha=1)
	handles, labels = plt.gca().get_legend_handles_labels()
	labels.reverse()
	handles.reverse()
	leg = plt.legend(loc=8,frameon=False,labels=labels,handles=handles,fontsize=8)
	for text in leg.get_texts():
		plt.setp(text, color = 'w')
	plt.margins(0,0)
	plt.ylabel('percentage of citations')
	plt.xlabel('publication year')
	ax1.tick_params(axis='x', which='major', pad=-5)
	ax1.tick_params(axis='y', which='major', pad=0)
	i,j,k,l = results[0]*100
	i,j,k,l = [i,(i+j),(i+j+k),(i+j+k+l)]
	i,j,k,l = [np.mean([0,i]),np.mean([i,j]),np.mean([j,k]),np.mean([k,l])]
	# i,j,k,l = np.array([100]) - np.array([i,j,k,l])  
	plt.sca(ax1)	
	ax1.yaxis.set_major_formatter(ticker.PercentFormatter()) 
	ax1.set_yticks([i,j,k,l])
	ax1.set_yticklabels(np.around(results[0]*100,0).astype(int))

	ax2 = ax1_plot[0].axes.twinx()
	plt.sca(ax2)
	i,j,k,l = results[-1]*100
	i,j,k,l = [i,(i+j),(i+j+k),(i+j+k+l)]
	i,j,k,l = [np.mean([0,i]),np.mean([i,j]),np.mean([j,k]),np.mean([k,l])] 
	plt.ylim(0,100)
	ax2.yaxis.set_major_formatter(ticker.PercentFormatter()) 
	ax2.set_yticks([i,j,k,l])
	ax2.set_yticklabels(np.around(results[-1]*100,0).astype(int))
	plt.xticks([1995., 2000., 2005., 2010., 2015., 2019],np.array([1995., 2000., 2005., 2010., 2015., 2019]).astype(int))   

	ax2.tick_params(axis='y', which='major', pad=0)
	plt.title('a',{'fontweight':'bold'},'left',pad=1)

	axes = []
	jidx = 3
	for makea in range(5):
		axes.append(fig.add_subplot(gs[jidx-3:jidx,6:10]))
		jidx=jidx+3
	
	for aidx,journal in enumerate(np.unique(df.journal)):
		results = np.zeros(( len(np.unique(df[(df.journal==journal)].year)),4))
		for yidx,year in enumerate(np.unique(df[(df.journal==journal)].year)):
			papers = df[(df.year==year)&(df.journal==journal)].index
			r = np.mean(plot_matrix[papers],axis=0).flatten()
			results[yidx,0] = r[0]
			results[yidx,1] = r[1]
			results[yidx,2] = r[2]
			results[yidx,3] = r[3]
		data = df[df.journal==journal]
		ax = axes[aidx]
		plt.sca(ax)
		ax1_plot = plt.stackplot(np.unique(data.year),results.transpose()*100, labels=labels,colors=pal, alpha=1)
		plt.margins(0,0)
		ax.set_yticks([])
		ax.set_xticks([])
		plt.title(journal.title(), pad=-40,color='w',fontsize=8)
		if aidx == 0: plt.text(0,1,'b',{'fontweight':'bold'},horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes)


	journals = np.unique(df.journal)
	plot_df = pd.DataFrame(columns=['journal','year','percentage','iteration'])  
	for j in journals:
		for yidx,year in enumerate(np.unique(df.year)):
			for i in range(n_iters):
				data = df[(df.year==year)&(df.journal==j)]
				papers = data.sample(int(len(data)),replace=True).index
				r = np.mean(plot_matrix[papers],axis=0).flatten()
				total = r.sum()
				r = np.array(r[1:])/total
				r = r.sum()
				tdf = pd.DataFrame(np.array([j,r,year,i]).reshape(1,-1),columns=['journal','percentage','year','iteration']) 
				plot_df = plot_df.append(tdf,ignore_index=True)


	plot_df.percentage = plot_df.percentage.astype(float)
	plot_df.iteration = plot_df.iteration.astype(int)
	plot_df.percentage = plot_df.percentage.astype(float) * 100
	pct_df = pd.DataFrame(columns=['journal','year','percentage','iteration'])
	plot_df = plot_df.sort_values('year')
	for i in range(n_iters):
		for j in journals:
			change = np.diff(plot_df[(plot_df.iteration==i)&(plot_df.journal==j)].percentage)
			tdf = pd.DataFrame(columns=['journal','year','percentage','iteration'])
			tdf.year = range(1997,2020)
			tdf.percentage = change[1:]
			tdf.journal = j
			tdf.iteration = i
			pct_df = pct_df.append(tdf,ignore_index=True)


	pct_df = pct_df.dropna()
	pct_df = pct_df[np.isinf(pct_df.percentage)==False] 
	mean_pc = pct_df.groupby(['journal']).mean() 
	min_pc = mean_pc - pct_df.groupby(['journal']).std() *2
	max_pc = mean_pc + pct_df.groupby(['journal']).std() *2  
	mean_pc = np.around(mean_pc.values,2)
	min_pc = np.around(min_pc.values,2)
	max_pc = np.around(max_pc.values,2)

	axes = []
	jidx = 3
	for makea in range(5):
		axes.append(fig.add_subplot(gs[jidx-3:jidx,10:]))
		jidx=jidx+3 

	for i,ax,journal,color in zip(range(5),axes,journals,sns.color_palette("rocket_r", 5)):
		plt.sca(ax)
		if i == 0: plt.title('c',{'fontweight':'bold'},'left',pad=1)
		lp = sns.lineplot(data=plot_df[plot_df.journal==journal],y='percentage',x='year',color=color,ci='sd')   
		plt.margins(0,0)
		ax.set_yticks([])
		ax.set_xticks([])
		plt.ylabel('')
		plt.xlabel('')
		thisdf = plot_df[plot_df.journal==journal]
		minp = int(np.around(thisdf.mean()['percentage'],0))
		thisdf = thisdf[thisdf.year==thisdf.year.max()]
		maxp = int(np.around(thisdf.mean()['percentage'],0))
		plt.text(0.01,.75,'%s'%(minp),horizontalalignment='left',verticalalignment='top', transform=ax.transAxes,fontsize=8)
		plt.text(1,.9,'%s'%(maxp),horizontalalignment='right',verticalalignment='top', transform=ax.transAxes,fontsize=8)
		plt.text(.99,0,'95%' + "CI: %s<%s>%s"%(min_pc[i][0],mean_pc[i][0],max_pc[i][0]),horizontalalignment='right',verticalalignment='bottom', transform=ax.transAxes,fontsize=8)
	plt.savefig('/%s/figures/figure1_pr_%s.pdf'%(homedir,method))

def make_pr_control():
	"""
	control for features of citing article
	"""
	# 1) the year of publication
	# 2) the journal in which it was published
	# 3) the number of authors
	# 4) whether the paper was a review article
	# 5) the seniority of the paper’s first and last authors.
	# 6) paper location
	df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	prs = np.load('/%s/data/result_pr_df_%s.npy'%(homedir,method))
	cont = pd.read_csv('/%s/article_data/CountryAndContData.csv'%(homedir)) 
	df = df.merge(cont,how='outer',left_index=True, right_index=True)
	df = df.merge(pd.read_csv('/%s/article_data/SeniorityData.csv'%(homedir)),left_index=True, right_index=True)

	reg_df = pd.DataFrame(columns=['year','n_authors','journal','paper_type','senior','location'])
	
	for entry in tqdm.tqdm(df.iterrows(),total=len(df)):
		idx = entry[0]
		paper = entry[1]
		year = entry[1].PY
		n_authors = len(paper.AF.split(';'))
		journal = entry[1].SO
		paper_type = paper.DT
		senior = entry[1].V4
		try: loc = entry[1]['FirstListed.Cont'].split()[0]
		except: loc = 'None'
		reg_df.loc[len(reg_df)] = [year,n_authors,journal,paper_type,senior,loc]

	reg_df["n_authors"] = pd.to_numeric(reg_df["n_authors"])
	reg_df["year"] = pd.to_numeric(reg_df["year"])
	reg_df["senior"] = pd.to_numeric(reg_df["senior"])
	
	skl_df = pd.get_dummies(reg_df).values

	ridge = MultiOutputRegressor(RidgeCV(alphas=[1e-5,1e-4,1e-3, 1e-2, 1e-1, 1,10,25,50,75,100])).fit(skl_df,prs.reshape(prs.shape[0],-1))
	ridge_probabilities = ridge.predict(skl_df)
	ridge_probabilities =  np.divide((ridge_probabilities), np.sum(ridge_probabilities,axis=1).reshape(-1,1))
	ridge_probabilities = ridge_probabilities.reshape(ridge_probabilities.shape[0],8,8)  

	np.save('/%s/data/probabilities_pr_%s.npy'%(homedir,method),ridge_probabilities)	

def make_control():
	"""
	control for features of citing article
	"""
	# 1) the year of publication
	# 2) the journal in which it was published
	# 3) the number of authors
	# 4) whether the paper was a review article
	# 5) the seniority of the paper’s first and last authors.
	# 6) paper location
	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	race_df = make_df()
	labels = ['white author & white author','white author & author of color',\
	'author of color & white author','author of color & author of color']
	df = race_df.merge(main_df,how='outer',left_index=True, right_index=True)
	journals = np.unique(df.journal)
	cont = pd.read_csv('/%s/article_data/CountryAndContData.csv'%(homedir)) 
	df = df.merge(cont,how='outer',left_index=True, right_index=True)
	df = df.merge(pd.read_csv('/%s/article_data/SeniorityData.csv'%(homedir)),left_index=True, right_index=True)

	matrix_idxs = {'white_M':0,'white_W':1,'white_U':2,'api_M':3,'api_W':4,'api_U':5,'hispanic_M':6,'hispanic_W':7,'hispanic_U':8,'black_M':9,'black_W':10,'black_U':11}

	df.fa_category = df.fa_category.map(matrix_idxs) 
	df.la_category = df.la_category.map(matrix_idxs)

	reg_df = pd.DataFrame(columns=['race','year','n_authors','journal','paper_type','senior','location'])
	
	for entry in tqdm.tqdm(df.iterrows(),total=len(df)):
		idx = entry[0]
		paper = entry[1]
		year = paper.year
		n_authors = len(paper.AF.split(';'))
		journal = paper.journal
		paper_type = paper.DT
		race = np.ravel_multi_index((paper.fa_category,paper.la_category),(12,12))
		senior = paper.V4
		try: loc = paper['FirstListed.Cont'].split()[0]
		except: loc = 'None'
		reg_df.loc[len(reg_df)] = [race,year,n_authors,journal,paper_type,senior,loc]

	reg_df["n_authors"] = pd.to_numeric(reg_df["n_authors"])
	reg_df["year"] = pd.to_numeric(reg_df["year"])
	reg_df["senior"] = pd.to_numeric(reg_df["senior"])
	reg_df["race"] = pd.to_numeric(reg_df["race"])

	skl_df = pd.get_dummies(reg_df).drop(columns=['race']).values
	clf = RandomForestClassifier().fit(skl_df,reg_df["race"])
	RFprobabilities = clf.predict_proba(skl_df)
	mlp = MLPClassifier(hidden_layer_sizes=(100,100,100,100,100))
	mlp.fit(skl_df,reg_df["race"])
	DNNprobabilities = mlp.predict_proba(skl_df)
	
	prs = (DNNprobabilities + RFprobabilities) /2
	
	assert clf.classes_ == mlp.classes_ 

	final_dnn_prs = np.zeros((df.shape[0],144))
	final_dnn_prs[:,mlp.classes_] = DNNprobabilities
	
	final_rf_prs = np.zeros((df.shape[0],144))
	final_rf_prs[:, clf.classes_] = RFprobabilities

	prs = final_rf_prs + final_dnn_prs /2
	np.save('/%s/data/probabilities_%s.npy'%(homedir,method),prs)	

def write_matrix():
	main_df = pd.read_csv('/%s/data/ArticleDataNew.csv'%(homedir))  
	df = pd.read_csv('/%s/data/result_df_%s.csv'%(homedir,method))

	df = df.dropna()

	if method =='wiki':
		df.fa_race = df.fa_race.map(wiki_2_race) 
		df.la_race = df.la_race.map(wiki_2_race) 

	citation_matrix = np.zeros((4,4))
	small_matrix = np.zeros((2,2))
	matrix_idxs = {'white':0,'api':1,'hispanic':2,'black':3}
	small_idxs = {'white':0,'api':1,'hispanic':1,'black':1}

	for fa_r in ['white','api','hispanic','black']:
		for la_r in ['white','api','hispanic','black']:
			citation_matrix[matrix_idxs[fa_r],matrix_idxs[la_r]] = len(df[(df.fa_race==fa_r)&(df.la_race==la_r)])
			small_matrix[small_idxs[fa_r],small_idxs[la_r]] += len(df[(df.fa_race==fa_r)&(df.la_race==la_r)])
	np.save('/%s/expected_matrix_%s.npy'%(homedir,method),citation_matrix)
	np.save('/%s/expected_small_matrix_%s.npy'%(homedir,method),small_matrix)

def make_pr_percentages():
	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	race_df = pd.read_csv('/%s/data/result_df_%s.csv'%(homedir,method))
	df = race_df.merge(main_df,how='outer',left_index=True, right_index=True)
	citing_prs = np.load('/%s/data/result_pr_df_%s.npy'%(homedir,method))
	
	if control == True: base_prs = np.load('/%s/data/probabilities_pr_%s.npy'%(homedir,method))
	
	elif control == 'null': base_prs = np.load('/%s/data/probabilities_pr_%s.npy'%(homedir,method))

	else:
		if walk_length == 'cited':
			base_prs = np.load('/%s/data/walk_pr_probabilities_%s_cited.npy'%(homedir,method)).reshape(-1,8,8)
		if walk_length[:3] == 'all':
			base_prs = np.load('/%s/data/walk_pr_probabilities_%s_%s.npy'%(homedir,method,walk_length)).reshape(-1,8,8)

	if control == 'null':
		matrix = np.zeros((100,df.shape[0],8,8))
		matrix[:] = np.nan

		base_matrix = np.zeros((100,df.shape[0],8,8))
		base_matrix[:] = np.nan
	else:
		matrix = np.zeros((df.shape[0],8,8))
		matrix[:] = np.nan

		base_matrix = np.zeros((df.shape[0],8,8))
		base_matrix[:] = np.nan

	if control == False:
		year_df = pd.DataFrame(columns=['year','month','prs'])
		for year in df.PY.unique():
			if year < 2009:continue
			print (year)
			for month in df.PD.unique():
				rdf = df[(df.year<year) | ((df.year==year) & (df.PD<=month))]
				this_base_matrix = citing_prs[rdf.index.values].mean(axis=0)
				year_df = year_df.append(pd.DataFrame(np.array([year,month,this_base_matrix]).reshape(1,-1),columns=['year','month','prs']),ignore_index=True) 


	for idx,paper in tqdm.tqdm(df.iterrows(),total=df.shape[0]):
		#only look at papers published 2009 or later
		year = paper.year
		if year < 2009:continue
		
		#only look at papers that cite at least 10 papers in our data
		if type(paper.CP) != str:
			if np.isnan(paper.CP)==True: continue
		n_cites = len(paper['CP'].split(','))
		if n_cites < 10: continue

		if control != 'null': this_matrix = citing_prs[np.array(paper['CP'].split(',')).astype(int)-1].sum(axis=0)
		
		if control == 'null':
			for i in range(100):
				this_base_matrix = []
				this_matrix = []
				for p in base_prs[np.array(paper['CP'].split(',')).astype(int)-1]:
					if np.min(p) < 0:p = p + abs(np.min(p))
					p = p + abs(np.min(p))
					p = p.flatten()/p.sum()
					this_base_matrix.append(p.reshape((8,8)))
					choice = np.zeros((8,8))
					choice[np.unravel_index(np.random.choice(range(64),p=p),(8,8))] = 1
					this_matrix.append(choice)
				this_base_matrix = np.sum(this_base_matrix,axis=0)
				this_matrix = np.sum(this_matrix,axis=0)
				matrix[i,idx] = this_matrix
				base_matrix[i,idx] = this_base_matrix

		else:
			if control == False:
				this_base_matrix = year_df[(year_df.year==year) & (year_df.month<=month)]['prs'].values[0]  * n_cites
			if control == True:
				this_base_matrix = base_prs[np.array(paper['CP'].split(',')).astype(int)-1].sum(axis=0)
			if control == 'walk':
				this_base_matrix = np.nansum(base_prs[np.array(paper['CP'].split(',')).astype(int)-1],axis=0)

			matrix[idx] = this_matrix
			base_matrix[idx] = this_base_matrix

	if type(control) == bool:
		np.save('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,control),matrix)
		np.save('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,control),base_matrix)
	elif control =='null':
		np.save('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,control),matrix)
		np.save('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,control),base_matrix)
	else:
		np.save('/%s/data/citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length),matrix)
		np.save('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length),base_matrix)
		
def make_percentages():
	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	race_df = pd.read_csv('/%s/data/result_df_%s.csv'%(homedir,method))
	df = race_df.merge(main_df,how='outer',left_index=True, right_index=True)
	if control == True: prs = np.load('/%s/data/probabilities_%s.npy'%(homedir,method)).reshape(-1,12,12)
	if control == 'walk': 
		if walk_length == 'cited':
			prs = np.load('/%s/data/walk_probabilities_%s_cited.npy'%(homedir,method)).reshape(-1,12,12)
		if walk_length == 'all':
			prs = np.load('/%s/data/walk_probabilities_%s_all.npy'%(homedir,method)).reshape(-1,12,12)
	
	matrix = np.zeros((df.shape[0],12,12))
	matrix[:] = np.nan

	base_matrix = np.zeros((df.shape[0],12,12))
	base_matrix[:] = np.nan

	if control == False:
		year_df = pd.DataFrame(columns=['year','month','prs'])
		for year in df.PY.unique():
			if year < 2009:continue
			print (year)
			for month in df.PD.unique():
				this_base_matrix = np.zeros((12,12))
				rdf = df[(df.year<year) | ((df.year==year) & (df.PD<=month))]
				for c1 in matrix_idxs.keys():
					for c2 in matrix_idxs.keys():
						this_base_matrix[matrix_idxs[c1],matrix_idxs[c2]] = len(rdf[(rdf.fa_category==c1)&(rdf.la_category==c2)])
				year_df = year_df.append(pd.DataFrame(np.array([year,month,this_base_matrix]).reshape(1,-1),columns=['year','month','prs']),ignore_index=True) 


	for idx,paper in tqdm.tqdm(df.iterrows(),total=df.shape[0]):
		#only look at papers published 2009 or later
		year = paper.year
		if year < 2009:continue
		
		#only look at papers that cite at least 10 papers in our data
		if type(paper.CP) != str:
			if np.isnan(paper.CP)==True: continue
		if len(paper['CP'].split(',')) < 10: continue
		#store a matrix where a 1 represents a paper between first and last author race / gender categories
		this_matrix = np.zeros((12,12))
		#store base rate, so, the number of papers in each category published before this paper
		this_base_matrix = np.zeros((12,12))
		#loop through the citations in the data
		for cidx,cited_paper in enumerate(paper['CP'].split(',')):
			cited_paper = df.loc[int(cited_paper)-1]
			this_matrix[matrix_idxs['%s_%s'%(cited_paper.fa_race,cited_paper.fa_g)],matrix_idxs['%s_%s'%(cited_paper.la_race,cited_paper.la_g)]] += 1

		if control == False:
			this_base_matrix = year_df[(year_df.year==year) & (year_df.month<=month)]['prs'].values[0]   
		if control == True:
			for cidx,cited_paper in enumerate(paper['CP'].split(',')):
				this_base_matrix = this_base_matrix + prs[int(cited_paper)-1]
		if control == 'walk':this_base_matrix = prs[idx]

	
		matrix[idx] = this_matrix
		base_matrix[idx] = this_base_matrix
	if type(control) == bool:
		np.save('/%s/data/citation_matrix_%s_%s.npy'%(homedir,method,control),matrix)
		np.save('/%s/data/base_citation_matrix_%s_%s.npy'%(homedir,method,control),base_matrix)
	else:
		np.save('/%s/data/citation_matrix_%s_%s_%s.npy'%(homedir,method,control,walk_length),matrix)
		np.save('/%s/data/base_citation_matrix_%s_%s_%s.npy'%(homedir,method,control,walk_length),base_matrix)
		
def plot_intersections():
	matrix = np.load('/%s/data/citation_matrix_%s_%s.npy'%(homedir,method,control))
	base_matrix = np.load('/%s/data/base_citation_matrix_%s_%s.npy'%(homedir,method,control))

	matrix_idxs = {'white_M':0,'white_W':1,'white_U':2,'api_M':3,'api_W':4,'api_U':5,'hispanic_M':6,'hispanic_W':7,'hispanic_U':8,'black_M':9,'black_W':10,'black_U':11}

	matrix = matrix[:,[0,1,3,4,6,7,9,10]][:,:,[0,1,3,4,6,7,9,10]]
	base_matrix = base_matrix[:,[0,1,3,4,6,7,9,10]][:,:,[0,1,3,4,6,7,9,10]]

	# boot_matrix = np.zeros((10000,8,8))
	# for b in range(10000):
	# 	print (b)
	# 	papers = np.random.choice(range(matrix.shape[0]),matrix.shape[0],replace=True)
	# 	m = np.nansum(matrix[papers],axis=0)
	# 	m = m / np.sum(m)
	# 	e = np.nansum(base_matrix[papers],axis=0) 
	# 	e = e / np.sum(e) 

	# 	rate = (m - e) / e
	# 	boot_matrix[b] = rate

	# np.save('/%s/data/intersection_boot_matrix.npy'%(homedir),boot_matrix)
		
	m = np.nansum(matrix,axis=0)
	m = m / np.sum(m)
	e = np.nansum(base_matrix,axis=0) 
	e = e / np.sum(e) 

	rate = (m - e) / e

	rate = rate * 100

	names = ['white_M','white_W','api_M','api_W','hispanic_M','hispanic_W','black_M','black_W']
	sns.set(style='whitegrid',font='Palatino')
	heat = sns.heatmap((rate).astype(int),annot=True,fmt='g',vmax=25,vmin=-25)
	heat.set_ylabel('first author',labelpad=0)  
	heat.set_yticklabels(names,rotation=25)
	heat.set_xlabel('last author',labelpad=0)  
	heat.set_xticklabels(names,rotation=65)
	heat.set_title('percentage over/under-citations')
	plt.tight_layout()  
	plt.savefig('/%s/figures/intersection_matrix_%s_%s.pdf'%(homedir,method,control))
	plt.close()


	orig_matrix = np.load('/%s/data/citation_matrix_%s_%s.npy'%(homedir,method,control))
	orig_base_matrix = np.load('/%s/data/base_citation_matrix_%s_%s.npy'%(homedir,method,control))
	race = 'black'
	df = pd.DataFrame(columns=['bias type','bias amount','boot','race'])
	for race in ['white','black','api','hispanic']:
		for idx in range(100):
			#norm matrix
			pick = np.random.choice(np.arange(orig_matrix.shape[0]),int(orig_matrix.shape[0]),replace=True)
			matrix = orig_matrix[pick]
			matrix = matrix / np.nansum(matrix)
			base_matrix= orig_base_matrix[pick]
			base_matrix = base_matrix / np.nansum(base_matrix)


			man_e1 = np.nansum(matrix[:,matrix_idxs['%s_M'%(race)],matrix_idxs['%s_M'%(race)]])
			man_b1 = np.nansum(base_matrix[:,matrix_idxs['%s_M'%(race)],matrix_idxs['%s_M'%(race)]])
			woman_e1 = np.nansum(matrix[:,matrix_idxs['%s_W'%(race)],matrix_idxs['%s_W'%(race)]])
			woman_b1 = np.nansum(base_matrix[:,matrix_idxs['%s_W'%(race)],matrix_idxs['%s_W'%(race)]])


			x =  ((man_e1 - man_b1)/ man_b1)  - ((woman_e1 - woman_b1)/ woman_b1)  # bias against women within this race

			if race == 'black':
				groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W','white_U','api_M','api_W','api_U','hispanic_M','hispanic_W','hispanic_U']),
				np.vectorize(matrix_idxs.get)(['black_M','black_W','black_U'])]

			if race == 'api':
				groups = [np.vectorize(matrix_idxs.get)(['white_U','white_M','white_W','hispanic_M','hispanic_W','hispanic_U','black_M','black_W','black_U']),
				np.vectorize(matrix_idxs.get)(['api_M','api_W','api_U'])]

			if race == 'hispanic':
				groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W','white_U','api_M','api_W','api_U','black_M','black_W','black_U']),
				np.vectorize(matrix_idxs.get)(['hispanic_M','hispanic_W','hispanic_U'])]

			if race == 'white':
				groups = [np.vectorize(matrix_idxs.get)(['hispanic_M','hispanic_W','hispanic_U','api_M','api_W','api_U','black_M','black_W','black_U']),
				np.vectorize(matrix_idxs.get)(['white_M','white_W','white_U'])]


			race_e1 = np.nansum(matrix[:,groups[1],groups[1]])
			race_b1 = np.nansum(base_matrix[:,groups[1],groups[1]])


			other_e1 = np.nansum(matrix[:,groups[0],groups[0]])

			other_b1 = np.nansum(base_matrix[:,groups[0],groups[0]])

			other = (other_e1 - other_b1) / other_b1
			race_c = (race_e1 - race_b1) / race_b1

			y = other - race_c # bias against this race
			df = df.append(pd.DataFrame(np.array(['gender',x,idx,race]).reshape(1,4),columns=['bias type','bias amount','boot','race']),ignore_index=True)
			df = df.append(pd.DataFrame(np.array(['race',y,idx,race]).reshape(1,4),columns=['bias type','bias amount','boot','race']),ignore_index=True)
		df['bias amount'] = df['bias amount'].astype(float)
		sns.boxenplot(data=df,y='bias amount',x='race',hue='bias type') 
		plt.ylabel('bias against women / race')
		plt.tight_layout()
		# plt.show()
		plt.savefig('/%s/figures/intersection_bars_%s_%s.pdf'%(homedir,method,control))
		plt.show()

		# x is the difference in citations between black men and black women
		# y is the difference in citation between black people and non-black people
		# what is bigger, x or y?

def plot_pr_intersections(control):
	n_iters = 10000
	if type(control) == bool:
		matrix = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
		base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
	else:
		matrix = np.load('/%s/data/citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length))
		base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length))

	null = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,'null'))
	null_base = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,'null'))[0]

	boot_matrix = np.zeros((n_iters,8,8))
	boot_r_matrix = np.zeros((n_iters,8,8))
	for b in range(n_iters):
		papers = np.random.choice(range(matrix.shape[0]),matrix.shape[0],replace=True)
		m = np.nansum(matrix[papers],axis=0)
		m = m / np.sum(m)
		e = np.nansum(base_matrix[papers],axis=0) 
		e = e / np.sum(e) 

		r = np.nansum(null[np.random.choice(100,1),papers],axis=0)
		r = r / np.sum(r)

		er = np.nansum(null_base[papers],axis=0) 
		er = er / np.sum(er) 


		rate = (m - e) / e
		r_rate = (r - er) / er
		boot_matrix[b] = rate
		boot_r_matrix[b] = r_rate

	# np.save('/%s/data/intersection_boot_matrix_%s.npy'%(homedir),boot_matrix,method)
		
	p_matrix = np.zeros((8,8))
	for i,j in combinations(range(8),2):
		x = boot_matrix[:,i,j]
		y = boot_r_matrix[:,i,j]
		p_matrix[i,j] = min(len(y[y>x.mean()]),len(y[y<x.mean()]))
	p_matrix = p_matrix / n_iters

	multi_mask = multipletests(p_matrix.flatten(),0.05,'fdr_by')[0].reshape(8,8) 

	names = ['white(m)','api(m)','hispanic(m)','black(m)','white(w)','api(w)','hispanic(w)','black(w)']



	# plt.tight_layout()  
	# plt.savefig('/%s/figures/intersection/intersection_matrix_%s_%s.pdf'%(homedir,method,control))
	# plt.close()


	if type(control) == bool:
		orig_matrix = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
		orig_base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
	else:
		orig_matrix = np.load('/%s/data/citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length))
		orig_base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length))



	df = pd.DataFrame(columns=['bias type','bias amount','boot','race'])
	for race in ['white','black','api','hispanic']:
		for idx in range(1000):
			#norm matrix
			pick = np.random.choice(np.arange(orig_matrix.shape[0]),int(orig_matrix.shape[0]),replace=True)
			matrix = orig_matrix[pick]
			matrix = matrix / np.nansum(matrix)
			base_matrix= orig_base_matrix[pick]
			base_matrix = base_matrix / np.nansum(base_matrix)


			man_e1 = np.nansum(matrix[:,matrix_idxs['%s_M'%(race)],matrix_idxs['%s_M'%(race)]])
			man_b1 = np.nansum(base_matrix[:,matrix_idxs['%s_M'%(race)],matrix_idxs['%s_M'%(race)]])
			woman_e1 = np.nansum(matrix[:,matrix_idxs['%s_W'%(race)],matrix_idxs['%s_W'%(race)]])
			woman_b1 = np.nansum(base_matrix[:,matrix_idxs['%s_W'%(race)],matrix_idxs['%s_W'%(race)]])


			x =  ((man_e1 - man_b1)/ man_b1)  - ((woman_e1 - woman_b1)/ woman_b1)  # bias against women within this race

			if race == 'black':
				groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W']),
				np.vectorize(matrix_idxs.get)(['black_M','black_W'])]

			if race == 'api':
				groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W']),
				np.vectorize(matrix_idxs.get)(['api_M','api_W'])]

			if race == 'hispanic':
				groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W']),
				np.vectorize(matrix_idxs.get)(['hispanic_M','hispanic_W'])]

			if race == 'white':
				groups = [np.vectorize(matrix_idxs.get)(['hispanic_M','hispanic_W','api_M','api_W','black_M','black_W']),
				np.vectorize(matrix_idxs.get)(['white_M','white_W'])]


			race_e1 = np.nansum(matrix[:,groups[1],groups[1]])
			race_b1 = np.nansum(base_matrix[:,groups[1],groups[1]])


			other_e1 = np.nansum(matrix[:,groups[0],groups[0]])

			other_b1 = np.nansum(base_matrix[:,groups[0],groups[0]])

			other = (other_e1 - other_b1) / other_b1
			race_c = (race_e1 - race_b1) / race_b1

			y = other - race_c # bias against this race
			df = df.append(pd.DataFrame(np.array(['gender',x,idx,race]).reshape(1,4),columns=['bias type','bias amount','boot','race']),ignore_index=True)
			df = df.append(pd.DataFrame(np.array(['race',y,idx,race]).reshape(1,4),columns=['bias type','bias amount','boot','race']),ignore_index=True)
	
	df['bias amount'] = df['bias amount'].astype(float) *  100

	plt.close()
	sns.set(style='whitegrid',font='Palatino')
	cmap = sns.diverging_palette(220, 10, as_cmap=True)
	fig = plt.figure(figsize=(7.5,3),constrained_layout=True)
	gs = gridspec.GridSpec(2, 2, figure=fig)
	ax1 = fig.add_subplot(gs[:2,:1])
	ax2 = fig.add_subplot(gs[:2,1:])
	plt.sca(ax1)
	d = np.around(np.nanmean(boot_matrix,axis=0)*100,0)
	d[multi_mask==False] = np.nan
	heat = sns.heatmap(d,annot=True,fmt='g',cmap=cmap,annot_kws={"size": 8})
	heat.set_ylabel('first author',labelpad=0)  
	heat.set_yticklabels(names,rotation=25)
	heat.set_xlabel('last author',labelpad=0)  
	heat.set_xticklabels(names,rotation=65)
	heat.set_title('a',{'fontweight':'bold'},'left',pad=1)


	# for text, show_annot in zip(ax1.texts, (element for row in multi_mask for element in row)):
		# text.set_visible(show_annot)

	cbar = heat.collections[0].colorbar
	cbar.ax.set_yticklabels(["{:.0%}".format(i/100) for i in cbar.get_ticks()])	
	plt.sca(ax2)
	df['bias amount'] = df['bias amount'].astype(float)*-1
	pal = [sns.diverging_palette(220, 10)[-2],sns.diverging_palette(220, 10)[1]]
	sns.barplot(data=df,y='bias amount',x='race',hue='bias type',palette=pal,ci='sd')
	# sns.boxenplot(data=df,y='bias amount',x='race',hue='bias type',palette=pal,saturation=1,cut=0,scale='width') 
	plt.ylabel('percent over-/under-citation')

	ax2.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax2.tick_params(axis='y', which='major', pad=-5)
	ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax2.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0)) 
	plt.legend(ncol=2,fontsize='small',frameon=False,columnspacing=0.5,handletextpad=0)
	plt.title('b',{'fontweight':'bold'},'left',pad=1)
	# plt.tight_layout()
	if type(control) == bool: plt.savefig('/%s/figures/intersection/intersection_%s_%s.pdf'%(homedir,method,control))
	else: plt.savefig('/%s/figures/intersection/intersection_%s_%s_%s.pdf'%(homedir,method,control,walk_length)) 
	plt.close()

def plot_ethnicolor_confusion():

	order = [] 
	for r in wiki_2_race.keys():
		order.append(r.split(',')[-1])
	r = [[873, 44, 7, 6, 6, 114, 8, 10, 7, 1, 8, 9, 6],
	 [17, 1300, 7, 20, 2, 58, 7, 6, 2, 0, 36, 10, 2],
	 [10, 10, 1188, 23, 107, 121, 21, 22, 15, 9, 17, 22, 7],
	 [5, 18, 48, 321, 72, 126, 12, 32, 31, 6, 37, 21, 5],
	 [6, 3, 118, 36, 824, 80, 45, 64, 23, 6, 15, 16, 12],
	 [52, 11, 57, 45, 52, 7341, 45, 260, 161, 39, 59, 101, 66],
	 [8, 5, 16, 14, 19, 84, 1262, 122, 21, 44, 18, 30, 23],
	 [7, 8, 27, 20, 66, 633, 119, 881, 59, 71, 80, 45, 32],
	 [13, 7, 14, 32, 34, 488, 37, 112, 1417, 41, 125, 118, 21],
	 [3, 0, 5, 7, 5, 167, 19, 98, 36, 318, 26, 23, 67],
	 [12, 12, 16, 19, 16, 174, 23, 56, 64, 18, 1437, 213, 22],
	 [4, 10, 13, 25, 8, 165, 34, 39, 99, 24, 147, 1790, 16],
	 [10, 2, 3, 7, 13, 141, 30, 31, 18, 44, 13, 11, 640]]

	plt.close()
	heat = sns.heatmap(np.array(r),vmax=1000,annot=True,fmt='g',annot_kws={"size": 8})
	locs, labels = plt.yticks()  
	plt.yticks(locs,order,rotation=360,**{'fontsize':12},) 
	locs, labels = plt.xticks() 
	plt.xticks(locs,order,rotation=90,**{'fontsize':12}) 

	plt.ylabel('observed',**{'fontsize':12}) 
	plt.xlabel('predicted',**{'fontsize':12}) 
	plt.tight_layout()
	plt.savefig('/%s/confusion_wiki.pdf'%(homedir))
	
	plt.close()
	r = [[5743, 42, 796, 3490],[257, 1693, 218, 22649],[173,82,25118,7609],[694,1157, 2442, 27837]]
	order = ['api','black','hispanic','white']
	heat = sns.heatmap(np.array(r),vmax=20000,annot=True,fmt='g',annot_kws={"size": 10})
	locs, labels = plt.yticks()  
	plt.yticks(locs,order,rotation=360,**{'fontsize':12},) 
	locs, labels = plt.xticks() 
	plt.xticks(locs,order,rotation=90,**{'fontsize':12}) 
	plt.ylabel('observed',**{'fontsize':12}) 
	plt.xlabel('predicted',**{'fontsize':12}) 
	plt.tight_layout()
	plt.savefig('/%s/confusion_census.pdf'%(homedir))

def plot_pr_percentages_booty_matrix(func_vars):
	control,within_poc,walk_papers = func_vars[0],func_vars[1],func_vars[2]
	"""
	Figure 2
	"""

	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	race_df = pd.read_csv('/%s/data/result_df_%s.csv'%(homedir,method))
	df = race_df.merge(main_df,how='outer',left_index=True, right_index=True)
	

	null = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,'null'))
	null_base = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,'null'))

	if type(control) == bool:
		matrix = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
		base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
	else:
		matrix = np.load('/%s/data/citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length))
		base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s_%s.npy'%(homedir,method,control,walk_length))

	if walk_papers == True:
		walk_base_matrix = np.load('/%s/data/base_citation_matrix_%s_walk.npy'%(homedir,method))
		matrix[np.isnan(walk_base_matrix[:,0,0])] = np.nan
		base_matrix[np.isnan(walk_base_matrix[:,0,0])] = np.nan

	matrix_idxs = {'white_M':0,'api_M':1,'hispanic_M':2,'black_M':3,'white_W':4,'api_W':5,'hispanic_W':6,'black_W':7}

	if within_poc == False:
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W',]),
		np.vectorize(matrix_idxs.get)(['api_M','api_W','hispanic_M','hispanic_W','black_M','black_W',])]
		names = ['white-white','white-poc','poc-white','poc-poc']

	if within_poc == 'black':
		# groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W','api_M','api_W','hispanic_M','hispanic_W',]),
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W']),
		np.vectorize(matrix_idxs.get)(['black_M','black_W',])]
		names = ['white-white','white-black','black-white','black-black']

	if within_poc == 'api':
		# groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W','hispanic_M','hispanic_W','black_M','black_W',]),
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W']),
		np.vectorize(matrix_idxs.get)(['api_M','api_W',])]
		names = ['white-white','white-asian','asian-white','asian-asian']

	if within_poc == 'hispanic':
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W']),
		# groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W','api_M','api_W','black_M','black_W',]),
		np.vectorize(matrix_idxs.get)(['hispanic_M','hispanic_W',])]
		names = ['white-white','white-hispanic','hispanic-white','hispanic-hispanic']


	plot_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))
	plot_base_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))
	plot_null = np.zeros((null.shape[0],matrix.shape[0],len(groups),len(groups)))
	plot_null_base = np.zeros((null.shape[0],matrix.shape[0],len(groups),len(groups)))

	for i in range(len(groups)):
		for j in range(len(groups)):
			plot_matrix[:,i,j] = np.nansum(matrix[:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)
			plot_base_matrix[:,i,j] = np.nansum(base_matrix[:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)
			for iteration in range(null.shape[0]):
				plot_null[iteration,:,i,j] = np.nansum(null[iteration,:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)
				plot_null_base[iteration,:,i,j] = np.nansum(null_base[iteration,:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)


	#make sure that, if we don't have data for a paper, we also are not including it's base rates
	#this is mostly for when the random walk fails because it's not part of the graph.
	x = plot_matrix.sum(axis=1).sum(axis=1)
	y = plot_base_matrix.sum(axis=1).sum(axis=1)
	mask = np.where(x==0)[0] 
	assert y[mask].sum() == 0


	for papers in [df[df.year>=2009],df[(df.year>=2009)&(df.fa_race=='white')&(df.la_race=='white')],df[(df.year>=2009)&((df.fa_race!='white')|(df.la_race!='white'))]]:
		print (papers.citation_count.sum())
		sum_cites = papers.citation_count.sum()
		papers = papers.index
		emperical = np.nanmean(plot_matrix[papers],axis=0)
		expected = np.nanmean(plot_base_matrix[papers],axis=0)
		emperical = emperical / np.sum(emperical)
		expected = expected / np.sum(expected)
		rate = (emperical - expected) / expected
		p = np.array([np.around(emperical.flatten()*100,1),np.around(expected.flatten()*100,1)]).flatten()
		print ('Of the citations given between 2009 and 2019, WW papers received %s, compared to %s for WA papers, %s for AW papers, and %s for AA papers. The expected proportions based on the pool of citable papers were %s for WW, %s for WA, %s for AW, and %s for AA.'%(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]))
		p = np.around(rate.flatten()*100,1)
		print ('By this measure, WW papers were cited %s more than expected, WA papers were cited %s less than expected, AW papers were cited %s less than expected, and AA papers were cited %s less than expected.'%(p[0],p[1],p[2],p[3]))
		p = np.around(rate.flatten() * sum_cites,-2).astype(int)
		print ('These values correspond to WW papers being cited roughly %s more times than expected, compared to roughly %s more times for WA papers, %s fewer for AW papers, and %s fewer for AA papers'%(p[0],p[1],p[2],p[3]))


	n_iters = 10000

	data_type = np.zeros((4)).astype(str)
	data_type[:] = 'real'
	rdata_type = np.zeros((4)).astype(str)
	rdata_type[:] = 'random'


	data = pd.DataFrame(columns=['citation_rate','citation_type','data_type'])
	papers = df[df.year>=2009]
	for boot in range(n_iters):
		boot_papers = papers.sample(len(papers),replace=True).index
		
		emperical = np.nanmean(plot_matrix[boot_papers],axis=0)		
		expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
		emperical = emperical / np.sum(emperical)
		expected = expected / np.sum(expected)
		rate = (emperical - expected) / expected
		


		random = np.nanmean(plot_null[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
		e_random = np.nanmean(plot_null_base[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
		random = random / np.sum(random)
		e_random = e_random / np.sum(e_random)
		r_rate = (random - e_random) / e_random

		data = data.append(pd.DataFrame(data= np.array([rate.flatten(),names,data_type]).swapaxes(0,1),columns=['citation_rate','citation_type','data_type']),ignore_index=True)
		data = data.append(pd.DataFrame(data= np.array([r_rate.flatten(),names,rdata_type]).swapaxes(0,1),columns=['citation_rate','citation_type','data_type']),ignore_index=True)   
	
	data.citation_rate = (data.citation_rate.astype(float)*100)
	p_vals = np.zeros((4))
	for idx,name in enumerate(names):
		x = data[(data.data_type=='real')&(data.citation_type==name)].citation_rate.values
		y = data[(data.data_type=='random')&(data.citation_type==name)].citation_rate.values
		p_vals[idx] = min(len(y[y>x.mean()]),len(y[y<x.mean()]))                 
	
	p_vals = p_vals / n_iters

	


	plot_data = data[data.data_type=='real']
	mean = plot_data.groupby('citation_type',sort=False).mean()
	std = plot_data.groupby('citation_type',sort=False).std()	

	plt.close()
	sns.set(style='whitegrid',font='Palatino')
	fig = plt.figure(figsize=(7.5,3),constrained_layout=True)
	gs = gridspec.GridSpec(12, 10, figure=fig)
	ax1 = fig.add_subplot(gs[:12,:5])
	plt.sca(ax1)	
	bx = sns.violinplot(data=plot_data,y='citation_rate',x='citation_type',palette=pal,order=names,saturation=1,cut=0,scale='width')
	for i,v in enumerate(bx.collections[::2]):
		v.set_color(pal[i])
	bx2 = sns.violinplot(data=data[data.data_type=='random'],y='citation_rate',x='citation_type',palette=pal,order=names,saturation=1,cut=0,scale='width',inner=None)
	for i,v in enumerate(bx2.collections[8:]):
		v.set_color([pal[i][0],pal[i][1],pal[i][2],.35])
	plt.ylabel("percent over-/under-citation",labelpad=0)
	plt.xlabel('')
	plt.title('a, all citers',{'fontweight':'bold'},'left',pad=1)
	ax1.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax1.tick_params(axis='y', which='major', pad=-5)
	ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax1.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
	for i in range(4):
		m,s = mean.values[i],std.values[i]
		loc = m + (s*3)
		low = np.around(m - (s*2),1)[0]
		high = np.around(m + (s*2),1)[0]
		m = np.around(m,1)[0]
		ax1.text(i,loc,'%s<%s>%s\n%s'%(low,m,high,log_p_value(p_vals[i])),horizontalalignment='center',fontsize=8)


	ax2 = fig.add_subplot(gs[0:6,5:])
	ax3 = fig.add_subplot(gs[6:,5:])
	
	plt.sca(ax2)


	data = pd.DataFrame(columns=['citation_rate','citation_type','data_type'])
	papers = df[(df.year>=2009)&(df.fa_race=='white')&(df.la_race=='white')]
	for boot in range(n_iters):
		boot_papers = papers.sample(len(papers),replace=True).index
		
		emperical = np.nanmean(plot_matrix[boot_papers],axis=0)		
		expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
		emperical = emperical / np.sum(emperical)
		expected = expected / np.sum(expected)
		rate = (emperical - expected) / expected
		


		random = np.nanmean(plot_null[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
		e_random = np.nanmean(plot_null_base[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
		random = random / np.sum(random)
		e_random = e_random / np.sum(e_random)
		r_rate = (random - e_random) / e_random


		data = data.append(pd.DataFrame(data= np.array([rate.flatten(),names,data_type]).swapaxes(0,1),columns=['citation_rate','citation_type','data_type']),ignore_index=True)
		data = data.append(pd.DataFrame(data= np.array([r_rate.flatten(),names,rdata_type]).swapaxes(0,1),columns=['citation_rate','citation_type','data_type']),ignore_index=True)   
	
	data.citation_rate = (data.citation_rate.astype(float)*100)
	p_vals = np.zeros((4))
	for idx,name in enumerate(names):
		x = data[(data.data_type=='real')&(data.citation_type==name)].citation_rate.values
		y = data[(data.data_type=='random')&(data.citation_type==name)].citation_rate.values
		p_vals[idx] = min(len(y[y>x.mean()]),len(y[y<x.mean()]))                       
	p_vals = p_vals / n_iters


	plot_data = data[data.data_type=='real']
	mean = plot_data.groupby('citation_type',sort=False).mean()
	std = plot_data.groupby('citation_type',sort=False).std()	

	plt.sca(ax2)	
	bx = sns.violinplot(data=plot_data,y='citation_rate',x='citation_type',palette=pal,order=names,saturation=1,cut=0,scale='width')
	for i,v in enumerate(bx.collections[::2]):
		v.set_color(pal[i])
	bx2 = sns.violinplot(data=data[data.data_type=='random'],y='citation_rate',x='citation_type',palette=pal,order=names,saturation=1,cut=0,scale='width',inner=None)
	for i,v in enumerate(bx2.collections[8:]):
		v.set_color([pal[i][0],pal[i][1],pal[i][2],.35])
	# plt.ylabel("percent over-/under-citation",labelpad=0)
	plt.xlabel('')
	plt.ylabel('')
	plt.title('b, white citers',{'fontweight':'bold'},'left',pad=1)
	ax2.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax2.tick_params(axis='y', which='major', pad=-5)
	ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax2.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
	for i in range(4):
		m,s = mean.values[i],std.values[i]
		loc = m + (s*3)
		low = np.around(m - (s*2),1)[0]
		high = np.around(m + (s*2),1)[0]
		m = np.around(m,1)[0]
		ax2.text(i,loc,'%s<%s>%s\n%s'%(low,m,high,log_p_value(p_vals[i])),horizontalalignment='center',fontsize=8)


	plt.sca(ax3)


	data = pd.DataFrame(columns=['citation_rate','citation_type','data_type'])
	papers = df[(df.year>=2009)&((df.fa_race!='white')|(df.la_race!='white'))]
	for boot in range(n_iters):
		boot_papers = papers.sample(len(papers),replace=True).index
		
		emperical = np.nanmean(plot_matrix[boot_papers],axis=0)		
		expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
		emperical = emperical / np.sum(emperical)
		expected = expected / np.sum(expected)
		rate = (emperical - expected) / expected
		


		random = np.nanmean(plot_null[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
		e_random = np.nanmean(plot_null_base[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
		random = random / np.sum(random)
		e_random = e_random / np.sum(e_random)
		r_rate = (random - e_random) / e_random

		data = data.append(pd.DataFrame(data= np.array([rate.flatten(),names,data_type]).swapaxes(0,1),columns=['citation_rate','citation_type','data_type']),ignore_index=True)
		data = data.append(pd.DataFrame(data= np.array([r_rate.flatten(),names,rdata_type]).swapaxes(0,1),columns=['citation_rate','citation_type','data_type']),ignore_index=True)   
	
	data.citation_rate = (data.citation_rate.astype(float)*100)
	p_vals = np.zeros((4))
	for idx,name in enumerate(names):
		x = data[(data.data_type=='real')&(data.citation_type==name)].citation_rate.values
		y = data[(data.data_type=='random')&(data.citation_type==name)].citation_rate.values
		p_vals[idx] = min(len(y[y>x.mean()]),len(y[y<x.mean()]))                     
	p_vals = p_vals / n_iters


	plot_data = data[data.data_type=='real']
	mean = plot_data.groupby('citation_type',sort=False).mean()
	std = plot_data.groupby('citation_type',sort=False).std()	
	
	bx = sns.violinplot(data=plot_data,y='citation_rate',x='citation_type',palette=pal,order=names,saturation=1,cut=0,scale='width')
	for i,v in enumerate(bx.collections[::2]):
		v.set_color(pal[i])
	bx2 = sns.violinplot(data=data[data.data_type=='random'],y='citation_rate',x='citation_type',palette=pal,order=names,saturation=1,cut=0,scale='width',inner=None)
	for i,v in enumerate(bx2.collections[8:]):
		v.set_color([pal[i][0],pal[i][1],pal[i][2],.35])
	# plt.ylabel("percent over-/under-citation",labelpad=0)
	plt.xlabel('')
	plt.ylabel('')
	plt.title('c, citers of color',{'fontweight':'bold'},'left',pad=1)
	ax3.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax3.tick_params(axis='y', which='major', pad=-5)
	ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax3.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
	for i in range(4):
		m,s = mean.values[i],std.values[i]
		loc = m + (s*3)
		low = np.around(m - (s*2),1)[0]
		high = np.around(m + (s*2),1)[0]
		m = np.around(m,1)[0]
		ax3.text(i,loc,'%s<%s>%s\n%s'%(low,m,high,log_p_value(p_vals[i])),horizontalalignment='center',fontsize=8)


	ylim = np.array([ax3.get_ylim(),ax2.get_ylim()]).min(),np.array([ax3.get_ylim(),ax2.get_ylim()]).max()
	plt.sca(ax3)
	plt.ylim(ylim)
	plt.sca(ax2)
	plt.ylim(ylim)

	if type(control) == bool: plt.savefig('/%s/figures/percentages/method-%s_control-%s_poc-%s_wp-%s.pdf'%(homedir,method,control,within_poc,walk_papers))
	else: plt.savefig('/%s/figures/percentages/method-%s_control-%s_poc-%s_wl-%s.pdf'%(homedir,method,control,within_poc,walk_length))
	plt.close()

	# return None
	"""
	temporal trends
	"""
	n_iters = 1000
	white_data = pd.DataFrame(columns=['citation_rate','citation_type','year','base_rate','emperical_rate','data_type','boot'])
	for year in range(2009,2020):
			papers = df[(df.year==year)&(df.fa_race=='white')&(df.la_race=='white')]
			for boot in range(n_iters):
				boot_papers = papers.sample(len(papers),replace=True).index

				emperical = np.nanmean(plot_matrix[boot_papers],axis=0)
				expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
				emperical = emperical / np.sum(emperical)
				expected = expected / np.sum(expected)
				rate = (emperical - expected) / expected

				random = np.nanmean(plot_null[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
				e_random = np.nanmean(plot_null_base[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
				random = random / np.sum(random)
				e_random = e_random / np.sum(e_random)
				r_rate = (random - e_random) / e_random

				boot_df = pd.DataFrame(data= np.array([rate.flatten(),names]).swapaxes(0,1),columns=['citation_rate','citation_type'])
				boot_df['year'] = year
				boot_df['base_rate'] = expected.flatten()
				boot_df['emperical_rate'] = emperical.flatten()
				boot_df['data_type'] = 'real'
				boot_df['boot'] = boot
				white_data = white_data.append(boot_df,ignore_index=True)   

				boot_df = pd.DataFrame(data= np.array([r_rate.flatten(),names]).swapaxes(0,1),columns=['citation_rate','citation_type'])
				boot_df['year'] = year
				boot_df['base_rate'] = e_random.flatten()
				boot_df['emperical_rate'] = random.flatten()
				boot_df['data_type'] = 'random'
				boot_df['boot'] = boot
				white_data = white_data.append(boot_df,ignore_index=True)   
		
	white_data = white_data.dropna()
	white_data.citation_rate = (white_data.citation_rate.astype(float)*100)
	white_data.base_rate = (white_data.base_rate .astype(float)*100)
	white_data.emperical_rate = (white_data.emperical_rate.astype(float)*100)

	slope_boot_df = pd.DataFrame(columns=['slope','data','citation_type'])
	for boot in range(n_iters):
		for name in names:
			real_slope = scipy.stats.linregress(white_data[(white_data.data_type=='real')&(white_data.citation_type==name)&(white_data.boot==boot)].citation_rate.values,range(11))[0] 
			random_slope = scipy.stats.linregress(white_data[(white_data.data_type=='random')&(white_data.citation_type==name)&(white_data.boot==boot)].citation_rate.values,range(11))[0] 
			slope_boot_df = slope_boot_df.append(pd.DataFrame(data= np.array([[real_slope,random_slope],['real','random'],[name,name]]).swapaxes(0,1),columns=['slope','data','citation_type']))

	slope_boot_df.slope=slope_boot_df.slope.astype(float)


	non_white_data = pd.DataFrame(columns=['citation_rate','citation_type','year','base_rate','emperical_rate','data_type','boot'])
	for year in range(2009,2020):
			papers = df[(df.year==year)&((df.fa_race!='white')|(df.la_race!='white'))]
			for boot in range(n_iters):
				boot_papers = papers.sample(len(papers),replace=True).index

				emperical = np.nanmean(plot_matrix[boot_papers],axis=0)
				expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
				emperical = emperical / np.sum(emperical)
				expected = expected / np.sum(expected)
				rate = (emperical - expected) / expected

				random = np.nanmean(plot_null[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
				e_random = np.nanmean(plot_null_base[np.random.choice(plot_null.shape[0]),boot_papers],axis=0)
				random = random / np.sum(random)
				e_random = e_random / np.sum(e_random)
				r_rate = (random - e_random) / e_random

				boot_df = pd.DataFrame(data= np.array([rate.flatten(),names]).swapaxes(0,1),columns=['citation_rate','citation_type'])
				boot_df['year'] = year
				boot_df['base_rate'] = expected.flatten()
				boot_df['emperical_rate'] = emperical.flatten()
				boot_df['data_type'] = 'real'
				boot_df['boot'] = boot
				non_white_data = non_white_data.append(boot_df,ignore_index=True)   

				boot_df = pd.DataFrame(data= np.array([r_rate.flatten(),names]).swapaxes(0,1),columns=['citation_rate','citation_type'])
				boot_df['year'] = year
				boot_df['base_rate'] = e_random.flatten()
				boot_df['emperical_rate'] = random.flatten()
				boot_df['data_type'] = 'random'
				boot_df['boot'] = boot
				non_white_data = non_white_data.append(boot_df,ignore_index=True)   
		
	non_white_data = non_white_data.dropna()
	non_white_data.citation_rate = (non_white_data.citation_rate.astype(float)*100)
	non_white_data.base_rate = (non_white_data.base_rate .astype(float)*100)
	non_white_data.emperical_rate = (non_white_data.emperical_rate.astype(float)*100)

	non_white_slope_boot_df = pd.DataFrame(columns=['slope','data','citation_type'])
	for boot in range(n_iters):
		for name in names:
			real_slope = scipy.stats.linregress(non_white_data[(non_white_data.data_type=='real')&(non_white_data.citation_type==name)&(non_white_data.boot==boot)].citation_rate.values,range(11))[0] 
			random_slope = scipy.stats.linregress(non_white_data[(non_white_data.data_type=='random')&(non_white_data.citation_type==name)&(non_white_data.boot==boot)].citation_rate.values,range(11))[0] 
			non_white_slope_boot_df = non_white_slope_boot_df.append(pd.DataFrame(data= np.array([[real_slope,random_slope],['real','random'],[name,name]]).swapaxes(0,1),columns=['slope','data','citation_type']))

	non_white_slope_boot_df.slope=non_white_slope_boot_df.slope.astype(float)

	plt.close()
	sns.set(style='whitegrid',font='Palatino')
	fig = plt.figure(figsize=(7.5,6),constrained_layout=True)
	gs = fig.add_gridspec(4, 4)
	
	ax1 = fig.add_subplot(gs[:2,:2])
	ax2 = fig.add_subplot(gs[:2,2:])

	ax3 = fig.add_subplot(gs[2,0])
	ax4 = fig.add_subplot(gs[2,1])
	ax5 = fig.add_subplot(gs[3,0])
	ax6 = fig.add_subplot(gs[3,1])

	ax7 = fig.add_subplot(gs[2,2])
	ax8 = fig.add_subplot(gs[2,3])
	ax9 = fig.add_subplot(gs[3,2])
	ax10 = fig.add_subplot(gs[3,3])

	plt.sca(ax1)
	sns.lineplot(x="year", y="citation_rate",hue="citation_type",data=white_data[white_data.data_type=='real'],ax=ax1,hue_order=names,ci='sd',palette=pal)
	plt.legend(labels=names,ncol=2,fontsize='small',frameon=False,columnspacing=0.5,handletextpad=0)#bbox_to_anchor=(0., 1.05))
	ax1.set_xlabel('')
	plt.title('a, white citers',{'fontweight':'bold'},'left',pad=1)
	ax1.set_ylabel('percent over-/under-citation',labelpad=0)
	ax1.tick_params(axis='x', which='major', pad=-5)
	ax1.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax1.tick_params(axis='y', which='major', pad=-5)
	ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax1.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax1.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
	plt.axhline(0, color="grey", clip_on=False,linestyle='--')
	plt.xlim(2009,2019) 
	for color,name in zip(pal,names):
		y_val=white_data[(white_data.data_type=='real')&(white_data.citation_type==name)&((white_data.year==2017)|(white_data.year==2018)|(white_data.year==2019))].citation_rate.max()
		x = slope_boot_df[(slope_boot_df.data=='real')&(slope_boot_df.citation_type==name)].slope.values
		y = slope_boot_df[(slope_boot_df.data=='random')&(slope_boot_df.citation_type==name)].slope.values
		p_val = min(len(y[y>x.mean()]),len(y[y<x.mean()]))  
		p_val = p_val/n_iters
		print (p_val)
		p_val = log_p_value(p_val)
		plt.text(2019,y_val,'slope=%s,%s'%(np.around(x.mean(),2),p_val),horizontalalignment='right',verticalalignment='bottom',fontsize=8,color=color)



	plt.sca(ax2)
	sns.lineplot(x="year", y="citation_rate",hue="citation_type",data=non_white_data[non_white_data.data_type=='real'],ax=ax2,hue_order=names,ci='sd',palette=pal)
	plt.legend(labels=names,ncol=2,fontsize='small',frameon=False,columnspacing=0.5,handletextpad=0)#,bbox_to_anchor=(0., 1.05))
	ax2.set_xlabel('')
	# plt.axhline(0, color="grey", clip_on=False,axes=ax2,linestyle='--')
	plt.title('b, citer of color',{'fontweight':'bold'},'left',pad=1)
	sns.despine()
	ax2.set_ylabel('percent over-/under-citation',labelpad=0)
	ax2.tick_params(axis='x', which='major', pad=-5)
	ax2.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax2.tick_params(axis='y', which='major', pad=-5)
	ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax2.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax2.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
	fig.text(0.00, 0.26, 'percentage of citations', va='center', rotation='vertical')
	plt.axhline(0, color="grey", clip_on=False,linestyle='--')   
	plt.xlim(2009,2019) 
	for color,name in zip(pal,names):
		y_val=non_white_data[(non_white_data.data_type=='real')&(non_white_data.citation_type==name)&((non_white_data.year==2017)|(non_white_data.year==2018)|(non_white_data.year==2019))].citation_rate.max()
		x = non_white_slope_boot_df[(non_white_slope_boot_df.data=='real')&(non_white_slope_boot_df.citation_type==name)].slope.values
		y = non_white_slope_boot_df[(non_white_slope_boot_df.data=='random')&(non_white_slope_boot_df.citation_type==name)].slope.values
		p_val = min(len(y[y>x.mean()]),len(y[y<x.mean()]))  
		p_val = p_val/n_iters
		print (p_val)
		p_val = log_p_value(p_val)
		plt.text(2019,y_val,'slope=%s,%s'%(np.around(x.mean(),2),p_val),horizontalalignment='right',verticalalignment='bottom',fontsize=8,color=color)


	ylim = np.array(np.array([ax1.get_ylim(),ax1.get_ylim()]).min(),np.array([ax2.get_ylim(),ax2.get_ylim()]).max())
	plt.sca(ax1)
	plt.ylim(ylim*1.1)
	plt.sca(ax2)
	plt.ylim(ylim*1.1)

	white_data = white_data[white_data.data_type=='real']
	non_white_data = non_white_data[non_white_data.data_type=='real']

	label = True
	for ax,citation_type,color in zip([ax3,ax4,ax5,ax6],white_data.citation_type.unique(),pal):
		plt.sca(ax)
		if label == True:
			plt.title('c, white citers',{'fontweight':'bold'},'left',pad=1)
			label = False
		tmp_ax0 = sns.lineplot(x="year", y="emperical_rate",data=white_data[white_data.citation_type==citation_type],ci='sd',color=color,marker='o')
		tmp_ax1 = sns.lineplot(x="year", y="base_rate",data=white_data[white_data.citation_type==citation_type],ci='sd',color='grey',marker='o')
		ax.set_xlabel('')
		# ax3.set_ylabel('percentage of citations',labelpad=0)
		sns.despine()
		ax.yaxis.set_major_locator(plt.MaxNLocator(6))
		ax.tick_params(axis='y', which='major', pad=-5)
		ax.tick_params(axis='x', which='major', bottom=False,top=False,labelbottom=False)
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
		ax.set_ylabel('')
		ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1)) 

	label = True
	for ax,citation_type,color in zip([ax7,ax8,ax9,ax10],non_white_data.citation_type.unique(),pal):
		plt.sca(ax)
		if label == True: 
			plt.title('d, citers of color',{'fontweight':'bold'},'left',pad=1)
			label = False
		tmp_ax0 = sns.lineplot(x="year", y="emperical_rate",data=non_white_data[non_white_data.citation_type==citation_type],ci='sd',color=color,marker='o')
		tmp_ax1 = sns.lineplot(x="year", y="base_rate",data=non_white_data[non_white_data.citation_type==citation_type],ci='sd',color='grey',marker='o')
		ax.set_xlabel('')
		# ax3.set_ylabel('percentage of citations',labelpad=0)
		sns.despine()
		ax.yaxis.set_major_locator(plt.MaxNLocator(6))
		ax.tick_params(axis='y', which='major', pad=-5)
		ax.tick_params(axis='x', which='major', bottom=False,top=False,labelbottom=False)
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
		ax.set_ylabel('')
		ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1)) 
	
	if type(control) == bool: plt.savefig('/%s/figures/temporal/method-%s_control-%s_poc-%s_wp-%s.pdf'%(homedir,method,control,within_poc,walk_papers))
	else: plt.savefig('/%s/figures/temporal/method-%s_control-%s_poc-%s_wl-%s.pdf'%(homedir,method,control,within_poc,walk_length))
	plt.close()

def plot_percentages_booty_matrix(func_vars):
	control,within_poc,walk_papers = func_vars[0],func_vars[1],func_vars[2]
	"""
	Figure 2
	"""

	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	race_df = pd.read_csv('/%s/data/result_df_%s.csv'%(homedir,method))
	df = race_df.merge(main_df,how='outer',left_index=True, right_index=True)
	
	if type(control) == bool:
		matrix = np.load('/%s/data/citation_matrix_%s_%s.npy'%(homedir,method,control))
		base_matrix = np.load('/%s/data/base_citation_matrix_%s_%s.npy'%(homedir,method,control))
	else:
		matrix = np.load('/%s/data/citation_matrix_%s_%s_%s.npy'%(homedir,method,control,walk_length))
		base_matrix = np.load('/%s/data/base_citation_matrix_%s_%s_%s.npy'%(homedir,method,control,walk_length))

	if walk_papers == True:
		walk_base_matrix = np.load('/%s/data/base_citation_matrix_%s_walk.npy'%(homedir,method))
		matrix[np.isnan(walk_base_matrix[:,0,0])] = np.nan
		base_matrix[np.isnan(walk_base_matrix[:,0,0])] = np.nan

	matrix_idxs = {'white_M':0,'white_W':1,'white_U':2,'api_M':3,'api_W':4,'api_U':5,'hispanic_M':6,'hispanic_W':7,'hispanic_U':8,'black_M':9,'black_W':10,'black_U':11}

	if within_poc == False:
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W','white_U']),
		np.vectorize(matrix_idxs.get)(['api_M','api_W','api_U','hispanic_M','hispanic_W','hispanic_U','black_M','black_W','black_U'])]
		names = ['white-white','white-poc','poc-white','poc-poc']

	if within_poc == 'black':
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W','white_U','api_M','api_W','api_U','hispanic_M','hispanic_W','hispanic_U']),
		np.vectorize(matrix_idxs.get)(['black_M','black_W','black_U'])]
		names = ['nb-nb','nb-black','black-nb','black-black']

	if within_poc == 'api':
		groups = [np.vectorize(matrix_idxs.get)(['white_U','white_M','white_W','hispanic_M','hispanic_W','hispanic_U','black_M','black_W','black_U']),
		np.vectorize(matrix_idxs.get)(['api_M','api_W','api_U'])]
		names = ['na-na','na-asian','asian-na','asian-asian']

	if within_poc == 'hispanic':
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W','white_U','api_M','api_W','api_U','black_M','black_W','black_U']),
		np.vectorize(matrix_idxs.get)(['hispanic_M','hispanic_W','hispanic_U'])]
		names = ['nh-nh','nh-hispanic','hispanic-nh','hispanic-hispanic']


	plot_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))
	plot_base_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))

	for i in range(len(groups)):
		for j in range(len(groups)):
			plot_matrix[:,i,j] = np.nansum(matrix[:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)
			plot_base_matrix[:,i,j] = np.nansum(base_matrix[:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)


	ci = 100
	n_iters = 10000



	data = pd.DataFrame(columns=['citation_rate','citation_type'])
	papers = df[df.year>=2009]
	for boot in range(n_iters):
		boot_papers = papers.sample(len(papers),replace=True).index
		
		emperical = np.nanmean(plot_matrix[boot_papers],axis=0)
		expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
		
		emperical = emperical / np.sum(emperical)
		expected = expected / np.sum(expected)

		rate = (emperical - expected) / expected

		data = data.append(pd.DataFrame(data= np.array([rate.flatten(),names]).swapaxes(0,1),columns=['citation_rate','citation_type']),ignore_index=True)   
	
	data.citation_rate = (data.citation_rate.astype(float)*100)

	mean = data.groupby('citation_type',sort=False).mean()
	std = data.groupby('citation_type',sort=False).std()	

	plt.close()
	sns.set(style='whitegrid',font='Palatino')
	# fig = plt.figure(constrained_layout=True)
	fig = plt.figure(figsize=(7.5,3),constrained_layout=True)
	gs = gridspec.GridSpec(12, 10, figure=fig)
	ax1 = fig.add_subplot(gs[:12,:5])
	plt.sca(ax1)	
	bx = sns.violinplot(data=data,y='citation_rate',x='citation_type',palette=pal,order=names,saturation=1,cut=0,scale='width')
	for i,v in enumerate(bx.collections[::2]):
		v.set_color(pal[i])
	plt.axhline(0, color="grey", clip_on=False,linestyle='--')
	plt.ylabel("percent over-/under-citation",labelpad=0)
	plt.xlabel('')
	plt.title('a, all citers',{'fontweight':'bold'},'left',pad=1)
	ax1.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax1.tick_params(axis='y', which='major', pad=-5)
	ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax1.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
	for i in range(4):
		m,s = mean.values[i],std.values[i]
		loc = m + (s*3)
		low = np.around(m - (s*2),2)[0]
		high = np.around(m + (s*2),2)[0]
		ax1.text(i,loc,'CI:%s,%s'%(low,high),horizontalalignment='center',fontsize=8)


	ax2 = fig.add_subplot(gs[0:6,5:])
	ax3 = fig.add_subplot(gs[6:,5:])
	
	plt.sca(ax2)
	data = pd.DataFrame(columns=['citation_rate','citation_type'])
	papers = df[(df.year>=2009)&(df.fa_race=='white')&(df.la_race=='white')]
	for boot in range(n_iters):
		boot_papers = papers.sample(len(papers),replace=True).index
		
		emperical = np.nanmean(plot_matrix[boot_papers],axis=0)
		expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
		
		emperical = emperical / np.sum(emperical)
		expected = expected / np.sum(expected)

		rate = (emperical - expected) / expected

		data = data.append(pd.DataFrame(data= np.array([rate.flatten(),names]).swapaxes(0,1),columns=['citation_rate','citation_type']),ignore_index=True)   
	
	data = data.dropna()
	data.citation_rate = (data.citation_rate.astype(float)*100)
	mean = data.groupby('citation_type',sort=False).mean()
	std = data.groupby('citation_type',sort=False).std()	

	bx = sns.violinplot(data=data,y='citation_rate',x='citation_type',axes=ax2,order=names,palette=pal,saturation=1,cut=0,scale='width')
	for i,v in enumerate(bx.collections[::2]):
		v.set_color(pal[i])
	plt.ylabel('') 
	plt.xlabel('')
	plt.title('b, white citers',{'fontweight':'bold'},'left',pad=1)
	# plt.ylabel("percent\nover-/under-citation",labelpad=0,fontsize=8.8)
	plt.axhline(0, color="grey", clip_on=False,axes=ax2,linestyle='--')
	ax2.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax2.tick_params(axis='y', which='major', pad=-5)
	ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax2.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0)) 
	fig.text(0.5, 0.51, 'percent over-/under-citation', va='center', rotation='vertical')  
	for i in range(4):
		m,s = mean.values[i],std.values[i]
		loc = m + (s*3)
		low = np.around(m - (s*2),2)[0]
		high = np.around(m + (s*2),2)[0]
		ax2.text(i,loc,'CI:%s,%s'%(low,high),horizontalalignment='center',fontsize=8)

	plt.sca(ax3)
	data = pd.DataFrame(columns=['citation_rate','citation_type'])
	papers = df[(df.year>2009)&((df.fa_race!='white')|(df.la_race!='white'))]
	for boot in range(n_iters):
		boot_papers = papers.sample(len(papers),replace=True).index
		
		emperical = np.nanmean(plot_matrix[boot_papers],axis=0)
		expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
		
		emperical = emperical / np.sum(emperical)
		expected = expected / np.sum(expected)

		rate = (emperical - expected) / expected

		data = data.append(pd.DataFrame(data= np.array([rate.flatten(),names]).swapaxes(0,1),columns=['citation_rate','citation_type']),ignore_index=True)   
	
	data = data.dropna()
	data.citation_rate = (data.citation_rate.astype(float)*100)
	mean = data.groupby('citation_type',sort=False).mean()
	std = data.groupby('citation_type',sort=False).std()	

	bx = sns.violinplot(data=data,y='citation_rate',x='citation_type',axes=ax3,order=names,palette=pal,saturation=1,cut=0,scale='width')
	for i,v in enumerate(bx.collections[::2]):
		v.set_color(pal[i])
	plt.ylabel('') 
	plt.xlabel('') 
	# plt.ylabel("percent\nover-/under-citation",labelpad=0,fontsize=8.8)
	plt.title('c, citer of color',{'fontweight':'bold'},'left',pad=1)
	plt.axhline(0, color="grey", clip_on=False,axes=ax3,linestyle='--')
	sns.despine(bottom=True) 
	ax3.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax3.tick_params(axis='y', which='major', pad=-5)
	ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax3.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0)) 
	
	for i in range(4):
		m,s = mean.values[i],std.values[i]
		loc = m + (s*3)
		low = np.around(m - (s*2),2)[0]
		high = np.around(m + (s*2),2)[0]
		ax3.text(i,loc,'CI:%s,%s'%(low,high),horizontalalignment='center',fontsize=8)

	if type(control) == bool: plt.savefig('/%s/figures/percentages/method-%s_control-%s_poc-%s_bootmatrix_wp-%s.pdf'%(homedir,method,control,within_poc,walk_papers))
	else: plt.savefig('/%s/figures/percentages/method-%s_control-%s_poc-%s_wl-%s.pdf'%(homedir,method,control,within_poc,walk_length))
	plt.close()
	# 1/0

	white_data = pd.DataFrame(columns=['citation_rate','citation_type','year','base_rate','emperical_rate'])
	for year in range(2009,2020):
			papers = df[(df.year==year)&(df.fa_race=='white')&(df.la_race=='white')]
			for boot in range(1000):
				boot_papers = papers.sample(len(papers),replace=True).index

				emperical = np.nanmean(plot_matrix[boot_papers],axis=0)
				expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
				
				emperical = emperical / np.sum(emperical)
				expected = expected / np.sum(expected)

				rate = (emperical - expected) / expected

				boot_df = pd.DataFrame(data= np.array([rate.flatten(),names]).swapaxes(0,1),columns=['citation_rate','citation_type'])
				boot_df['year'] = year
				boot_df['base_rate'] = expected.flatten()
				boot_df['emperical_rate'] = emperical.flatten()
				white_data = white_data.append(boot_df,ignore_index=True)   
		
	white_data = white_data.dropna()
	white_data.citation_rate = (white_data.citation_rate.astype(float)*100)
	white_data.base_rate = (white_data.base_rate .astype(float)*100)
	white_data.emperical_rate = (white_data.emperical_rate.astype(float)*100)


	non_white_data = pd.DataFrame(columns=['citation_rate','citation_type','year','base_rate','emperical_rate'])
	for year in range(2009,2020):
			papers = df[(df.year==year)&((df.fa_race!='white')|(df.la_race!='white'))]
			for boot in range(1000):
				boot_papers = papers.sample(len(papers),replace=True).index

				emperical = np.nanmean(plot_matrix[boot_papers],axis=0)
				expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
				
				emperical = emperical / np.sum(emperical)
				expected = expected / np.sum(expected)

				rate = (emperical - expected) / expected

				boot_df = pd.DataFrame(data= np.array([rate.flatten(),names]).swapaxes(0,1),columns=['citation_rate','citation_type'])
				boot_df['year'] = year
				boot_df['base_rate'] = expected.flatten()
				boot_df['emperical_rate'] = emperical.flatten()
				non_white_data = non_white_data.append(boot_df,ignore_index=True)   
		
	non_white_data = non_white_data.dropna()
	non_white_data.citation_rate = (non_white_data.citation_rate.astype(float)*100)
	non_white_data.base_rate = (non_white_data.base_rate .astype(float)*100)
	non_white_data.emperical_rate = (non_white_data.emperical_rate.astype(float)*100)
	

	plt.close()
	sns.set(style='white',font='Palatino')
	fig = plt.figure(figsize=(7.5,6),constrained_layout=True)
	gs = fig.add_gridspec(4, 4)
	
	ax1 = fig.add_subplot(gs[:2,:2])
	ax2 = fig.add_subplot(gs[:2,2:])

	ax3 = fig.add_subplot(gs[2,0])
	ax4 = fig.add_subplot(gs[2,1])
	ax5 = fig.add_subplot(gs[3,0])
	ax6 = fig.add_subplot(gs[3,1])

	ax7 = fig.add_subplot(gs[2,2])
	ax8 = fig.add_subplot(gs[2,3])
	ax9 = fig.add_subplot(gs[3,2])
	ax10 = fig.add_subplot(gs[3,3])

	plt.sca(ax1)
	sns.lineplot(x="year", y="citation_rate",hue="citation_type",data=white_data,ax=ax1,hue_order=names,ci=ci,palette=pal)
	plt.legend(labels=names,ncol=4,loc=2,fontsize='x-small',frameon=False,columnspacing=0.5,handletextpad=0,bbox_to_anchor=(0., 1.05))
	ax1.set_xlabel('')
	plt.title('a, white citers',{'fontweight':'bold'},'left',pad=1)
	ax1.set_ylabel('percent over-/under-citation',labelpad=0)
	ax1.tick_params(axis='x', which='major', pad=-5)
	ax1.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax1.tick_params(axis='y', which='major', pad=-5)
	ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax1.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax1.yaxis.set_major_formatter(ticker.PercentFormatter()) 

	plt.sca(ax2)
	sns.lineplot(x="year", y="citation_rate",hue="citation_type",data=non_white_data,ax=ax2,hue_order=names,ci=ci,palette=pal)
	plt.legend(labels=names,ncol=4,loc=2,fontsize='x-small',frameon=False,columnspacing=0.5,handletextpad=0,bbox_to_anchor=(0., 1.05))
	ax2.set_xlabel('')
	# plt.axhline(0, color="grey", clip_on=False,axes=ax2,linestyle='--')
	plt.title('b, citer of color',{'fontweight':'bold'},'left',pad=1)
	sns.despine()
	ax2.set_ylabel('percent over-/under-citation',labelpad=0)
	ax2.tick_params(axis='x', which='major', pad=-5)
	ax2.yaxis.set_major_locator(plt.MaxNLocator(8))
	ax2.tick_params(axis='y', which='major', pad=-5)
	ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax2.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	ax2.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
	fig.text(0.01, 0.26, 'percentage of citations', va='center', rotation='vertical')     

	label = True
	for ax,citation_type,color in zip([ax3,ax4,ax5,ax6],white_data.citation_type.unique(),pal):
		plt.sca(ax)
		if label == True:
			plt.title('c, white citers',{'fontweight':'bold'},'left',pad=1)
			label = False
		tmp_ax0 = sns.lineplot(x="year", y="emperical_rate",data=white_data[white_data.citation_type==citation_type],ci=ci,color=color,marker='o')
		tmp_ax1 = sns.lineplot(x="year", y="base_rate",data=white_data[white_data.citation_type==citation_type],ci=ci,color='grey',marker='o')
		ax.set_xlabel('')
		# ax3.set_ylabel('percentage of citations',labelpad=0)
		sns.despine()
		ax.yaxis.set_major_locator(plt.MaxNLocator(6))
		ax.tick_params(axis='y', which='major', pad=-5)
		ax.tick_params(axis='x', which='major', bottom=False,top=False,labelbottom=False)
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
		ax.set_ylabel('')
		ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0)) 

	label = True
	for ax,citation_type,color in zip([ax7,ax8,ax9,ax10],non_white_data.citation_type.unique(),pal):
		plt.sca(ax)
		if label == True: 
			plt.title('d, citers of color',{'fontweight':'bold'},'left',pad=1)
			label = False
		tmp_ax0 = sns.lineplot(x="year", y="emperical_rate",data=non_white_data[non_white_data.citation_type==citation_type],ci=ci,color=color,marker='o')
		tmp_ax1 = sns.lineplot(x="year", y="base_rate",data=non_white_data[non_white_data.citation_type==citation_type],ci=ci,color='grey',marker='o')
		ax.set_xlabel('')
		# ax3.set_ylabel('percentage of citations',labelpad=0)
		sns.despine()
		ax.yaxis.set_major_locator(plt.MaxNLocator(6))
		ax.tick_params(axis='y', which='major', pad=-5)
		ax.tick_params(axis='x', which='major', bottom=False,top=False,labelbottom=False)
		ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
		ax.set_ylabel('')
		ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0)) 
	plt.savefig('/%s/figures/temporal/method-%s_control-%s_poc-%s_wp-%s.pdf'%(homedir,method,control,within_poc,walk_papers))
	plt.close()

def plot_all():
	global walk_length
	plot_pr_percentages_booty_matrix([False,False,False]) 
	plot_pr_percentages_booty_matrix([True,False,False]) 
	walk_length ='cited'
	plot_pr_percentages_booty_matrix(['walk',False,False]) 
	walk_length ='all'
	plot_pr_percentages_booty_matrix(['walk',False,False]) 
	for race in ['black','hispanic','api']:
		plot_pr_percentages_booty_matrix([False,race,False]) 
		plot_pr_percentages_booty_matrix([True,race,False]) 
		walk_length ='cited'
		plot_pr_percentages_booty_matrix(['walk',race,False]) 
		walk_length ='all'
		plot_pr_percentages_booty_matrix(['walk',race,False]) 
	# plot_pr_intersections(False)
	# plot_pr_intersections(True)
	# walk_length ='all'
	# plot_pr_intersections('walk')
	# walk_length ='cited'
	# plot_pr_intersections('walk')

def make_networks():
	"""
	co-author network, authors are nodes, co-authorship is an edge
	"""
	author_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	paper_df = pd.read_csv('/%s/data/all_data_%s.csv'%(homedir,method))
	author_df['races'] = paper_df.all_races.values

	g = igraph.Graph()
	node_2_a = {}
	a_2_node = {}
	a_2_paper = {}
	node_idx = 0
	#add authors to graph
	edges = []
	years = []
	races = []
	for p in tqdm.tqdm(author_df.iterrows(),total=len(author_df)):
		#get authors of this paper
		authors = p[1].AF.split('; ')
		p_races = p[1].races.split('_')
		for ai,a in enumerate(authors):
			#authors on more than one paper, so skip if already in graph
			if a in a_2_node.keys():continue
			# store papers for each author, used for edges later
			if a in a_2_paper.keys(): a_2_paper[a.strip()] = a_2_paper[a.strip()].append(p[0])
			else: a_2_paper[a.strip()] = [p[0]]
			#store index and author name
			node_2_a[node_idx] = a.strip()
			a_2_node[a.strip()] = node_idx
			#add author and race
			g.add_vertex(node_idx)
			races.append(p_races[ai])
			#on to the next node/author
			node_idx += 1
		#add edges between authors
		year = p[1].PY
		for co_a_i in p[1].AF.split('; '):
			for co_a_j in p[1].AF.split('; '):
				#look up nodal_index of co_author
				# 1/0
				nodes = [a_2_node[co_a_i.strip()],a_2_node[co_a_j.strip()]]
				i_node,j_node = np.min(nodes),np.max(nodes)
				edge = (i_node,j_node)
				edges.append(edge)
				years.append(year)
	edges,year_idx = np.unique(edges,axis=0,return_index=True) 
	years = np.array(years)[np.array(year_idx)]   
	g.add_edges(edges) 
	g.es['year'] = years
	g.vs['race'] = races
				
	g.write_pickle('/%s/data/%s_coa_graph'%(homedir,method))
	np.save('/%s/data/%s_a_2_node.npy'%(homedir,method), a_2_node) 
	np.save('/%s/data/%s_a_2_paper.npy'%(homedir,method), a_2_paper) 
	np.save('/%s/data/%s_node_2_a.npy'%(homedir,method), node_2_a) 

def write_graph():
	# for year in np.unique(author_df.PY.values):
	g = igraph.load('/%s/data/%s_coa_graph'%(homedir,method),format='pickle')  
	g.es.select(year_ne=2019).delete()
	g = g.clusters().giant()
	vc = g.community_fastgreedy().as_clustering(4)


	membership = vc.membership
	from matplotlib.colors import rgb2hex
	color_array = np.zeros((g.vcount())).astype(str)
	colors = np.array([[72,61,139],[82,139,139],[180,205,205],[205,129,98]])/255.
	for i in range(g.vcount()):
		color_array[i] = rgb2hex(colors[membership[i]])
	g.vs['color'] = color_array.astype('str')
	g.vs['sp'] = g.shortest_paths(18)[0]

	max_path = np.max(g.vs['sp'][0])	

	def walk(i):
		walked = np.zeros(g.vcount())
		walk_2 = 18
		for i in range(max_path*2):
			walk_2 = np.random.choice(g.neighbors(walk_2))
			walked[walk_2] += 1
		return walked
	pool = multiprocessing.Pool(8)
	walked = pool.map(walk,range(100000))

	sum_walk = np.sum(walked,axis=0)
	final_walk = np.zeros((g.vcount()))

	final_walk[sum_walk>0] = np.argsort(sum_walk[sum_walk>0]) 
	# walked[100] = 0

	g.vs['walks'] = final_walk
	g.write_gml('/%s/citation_network_%s_big.gml'%(homedir,method))

	from igraph import VertexClustering
	race_vs = VertexClustering.FromAttribute(g,'race')                                                                                                 

	print (vcc.modularity)
	print (race_vs.modularity)

def analyze_coa_mod():

	asian = [0,1,2]
	black = [3,4]
	white = [5,6,7,8,9,11,12]
	hispanic = [10]

	r = pd.read_csv('/%s/data/result_df_%s_all.csv'%(homedir,method))
	r.name = r.name.str.strip()
	node_2_a = np.load('/%s/data/%s_node_2_a.npy'%(homedir,method),allow_pickle='TRUE').item()
	g = igraph.load('/%s/data/%s_coa_graph'%(homedir,method),format='pickle')

	race_prs = np.zeros((g.vcount(),4))
	for node in tqdm.tqdm(range(g.vcount()),total=g.vcount()):
		prs = r[r.name==node_2_a[node].strip()].values[0][4:]  
		race_prs[node] = [np.sum(prs[white]),np.sum(prs[asian]),np.sum(prs[hispanic]),np.sum(prs[black])]

	race_prs = race_prs / np.sum(race_prs,axis=1)[:,None]  
	g.race_prs = race_prs

	def mod(year):
		yg = g.copy()
		yg.es.select(year_gt=year).delete()
		race_binary = np.zeros((g.vcount()))
		for node in range(g.vcount()):
			race_binary[node] = np.random.choice([1,2,3,4],p=g.race_prs[node])
		g.vs['race'] = race_binary
		yg = yg.clusters().giant()

		rm = VertexClustering.FromAttribute(yg,'race').modularity
		em = yg.community_infomap().modularity 
		return ([year,rm,em])

	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	q_vals = []
	for i in tqdm.tqdm(range(100),total=100):
		q_vals.append(pool.map(mod,np.arange(1995,2020)))

	years = []
	races = []
	emperical = []
	for i in q_vals:
		for q in i:
			years.append(q[0])
			races.append(q[1])
			emperical.append(q[2])


	np.save('/%s/data/q_analysis_emp_%s.npy'%(homedir,method),emperical)
	np.save('/%s/data/q_analysis_year_%s.npy'%(homedir,method),years)
	np.save('/%s/data/q_analysis_race_%s.npy'%(homedir,method),races)

def plot_coa_mod():
	emperical = np.load('/%s/data/q_analysis_emp_%s.npy'%(homedir,method))
	years = np.load('/%s/data/q_analysis_year_%s.npy'%(homedir,method))
	races = np.load('/%s/data/q_analysis_race_%s.npy'%(homedir,method))

	race_df = pd.DataFrame(np.array([races,years]).transpose(),columns=['q','year'])     
	race_df['network partition'] = 'race'
	e_df = pd.DataFrame(np.array([emperical,years]).transpose(),columns=['q','year'])     
	e_df['network partition'] = 'q-max'
	df = race_df.append(e_df,ignore_index=True)

	sns.set(style='white',font='Palatino')
	plt.close()
	fig = plt.figure(figsize=(3,2.5),constrained_layout=True)
	# y1 = sns.lineplot(np.unique(g.es['year']),(race-emperical)/emperical)  
	y1=sns.lineplot(years,races,color='salmon',label='race',legend=False,ci='sd',n_boot=1000)
	

	plt.ylabel('Q, race partition',color='salmon')
	y2 = y1.axes.twinx()
	plt.sca(y2)
	# sns.lineplot(np.unique(g.es['year']),race)  
	sns.lineplot(years,emperical,color='black',label='emperical',ci='sd',legend=False,n_boot=1000)
	plt.ylabel('Q, emperical partition')
	# y2.legend(loc=7)
	# y1.figure.legend(loc=3)
	y1.set_xticks([1995,2003,2011,2020]) 
	plt.savefig('q_analysis.pdf')

def multi_short(paper):
	global paper_df
	global main_df
	global graphs
	global node_2_a
	global a_2_node
	global a_2_paper
	global walk_length
	global prs
	print (paper)
	this_paper_df = paper_df.iloc[paper]
	this_paper_main_df = main_df.iloc[paper]
	citing_authors = [this_paper_main_df.AF.split(';')[0].strip(),this_paper_main_df.AF.split(';')[-1].strip()]
	year = this_paper_df.year
	yg,m,big = graphs[year]
	# get authors who are cited
	cited = this_paper_main_df.CP.split(', ')
	cited_authors = []
	for c in cited:
		cited_df = main_df.iloc[int(c)-1]
		for ca in cited_df.AF.split('; '):
			cited_authors.append(ca.strip())

		
	cited_authors = np.unique(cited_authors).flatten()
	cited_authors = np.setdiff1d(cited_authors,citing_authors) 
	# get shortest paths to papers cited
	all_paths = []
	cited_paths= []
	for i in citing_authors:
		i = a_2_node[i]
		all_paths.append(yg.shortest_paths(i,np.where(m==big)[0]))
		js = []
		for j in cited_authors: 
			j = a_2_node[j]
			js.append(j)
		cited_paths.append(yg.shortest_paths(i,js))
	cited_paths = np.array(cited_paths).reshape(1,-1)
	cited_paths = cited_paths[np.isinf(cited_paths)==False] 

	all_paths = np.array(all_paths).reshape(1,-1)
	all_paths = all_paths[np.isinf(all_paths)==False] 

	null_walks = 0

	base_matrix = np.zeros((12,12))
	base_matrix[:] = 0

	#if we take random walks, the length of the longest random path between two nodes, what is the base rate?
	# we only count times when the walk ends on a first or last author and paper was published before this paper
	if walk_length == 'cited':
		wl = int(np.nanmax(cited_paths))
	if walk_length == 'all':
		wl = int(np.nanmax(all_paths))
	if wl == 0:
		base_matrix[:] = np.nan
		return [paper,base_matrix,np.mean(cited_paths),np.mean(all_paths)]
	for i in citing_authors:
		i = a_2_node[i.strip()]
		while True:
			walk_2 = i # start from this author
			for w in range(wl):
				walk_2 = np.random.choice(yg.neighbors(walk_2),1)[0]
			cited_author = node_2_a[walk_2]
			walk_papers = a_2_paper[cited_author]
			np.random.shuffle(walk_papers)
			for walk_paper in walk_papers:
				walk_paper = paper_df.iloc[walk_paper]
				if walk_paper.year>= year: continue
				#you found a paper, store it in matrix! 
				base_matrix[matrix_idxs['%s_%s'%(walk_paper.fa_race,walk_paper.fa_g)],matrix_idxs['%s_%s'%(walk_paper.la_race,walk_paper.la_g)]] += 1
				null_walks += 1

			if null_walks > 1000:break
	return [paper,base_matrix,np.mean(cited_paths),np.mean(all_paths)]

def multi_pr_short(paper):
	global paper_df
	global main_df
	global graphs
	global node_2_a
	global a_2_node
	global a_2_paper
	global walk_length
	global prs
	print (paper)
	this_paper_df = paper_df.iloc[paper]
	this_paper_main_df = main_df.iloc[paper]
	citing_authors = [this_paper_main_df.AF.split(';')[0].strip(),this_paper_main_df.AF.split(';')[-1].strip()]
	year = this_paper_df.year
	yg,m,big = graphs[year]
	# get authors who are cited
	cited = this_paper_main_df.CP.split(', ')
	cited_authors = []
	for c in cited:
		cited_df = main_df.iloc[int(c)-1]
		for ca in cited_df.AF.split('; '):
			cited_authors.append(ca.strip())

		
	cited_authors = np.unique(cited_authors).flatten()
	cited_authors = np.setdiff1d(cited_authors,citing_authors) 
	# get shortest paths to papers cited
	all_paths = []
	cited_paths= []
	for i in citing_authors:
		i = a_2_node[i]
		all_paths.append(yg.shortest_paths(i,np.where(m==big)[0]))
		js = []
		for j in cited_authors: 
			j = a_2_node[j]
			js.append(j)
		cited_paths.append(yg.shortest_paths(i,js))
	cited_paths = np.array(cited_paths).reshape(1,-1)
	cited_paths = cited_paths[np.isinf(cited_paths)==False] 

	all_paths = np.array(all_paths).reshape(1,-1)
	all_paths = all_paths[np.isinf(all_paths)==False] 

	null_walks = 0

	base_matrix = np.zeros((8,8))
	base_matrix[:] = 0

	#if we take random walks, the length of the longest random path between two nodes, what is the base rate?
	# we only count times when the walk ends on a first or last author and paper was published before this paper
	if walk_length == 'cited':
		wl = int(np.nanmax(cited_paths))
	if walk_length == 'all':
		wl = int(np.nanmax(all_paths))
	if wl == 0:
		base_matrix[:] = np.nan
		return [paper,base_matrix,np.mean(cited_paths),np.mean(all_paths)]
	for i in citing_authors:
		i = a_2_node[i.strip()]
		while True:
			walk_2 = i # start from this author
			for w in range(wl):
				walk_2 = np.random.choice(yg.neighbors(walk_2),1)[0]
			cited_author = node_2_a[walk_2]
			walk_papers = a_2_paper[cited_author]
			np.random.shuffle(walk_papers)
			for walk_paper in walk_papers:
				walk_paper_df = paper_df.iloc[walk_paper]
				if walk_paper_df.year>= year: continue
				#you found a paper, store it in matrix! 
				base_matrix = base_matrix + prs[walk_paper]
				null_walks += 1

			if null_walks > 1000:break
	return [paper,base_matrix,np.mean(cited_paths),np.mean(all_paths)]

def multi_pr_shortv2(paper):
	global paper_df
	global main_df
	global graphs
	global node_2_a
	global a_2_node
	global a_2_paper
	global walk_length
	global prs
	print (paper)
	this_paper_df = paper_df.iloc[paper]
	this_paper_main_df = main_df.iloc[paper]
	citing_authors = [this_paper_main_df.AF.split(';')[0].strip(),this_paper_main_df.AF.split(';')[-1].strip()]
	year = this_paper_df.year
	yg,m,big = graphs[year]
	# get authors who are cited
	cited = this_paper_main_df.CP.split(', ')
	cited_authors = []
	for c in cited:
		cited_df = main_df.iloc[int(c)-1]
		for ca in cited_df.AF.split('; '):
			cited_authors.append(ca.strip())

		
	cited_authors = np.unique(cited_authors).flatten()
	cited_authors = np.setdiff1d(cited_authors,citing_authors) 
	# get shortest paths to papers cited
	all_paths = []
	cited_paths= []
	for i in citing_authors:
		i = a_2_node[i]
		all_paths.append(yg.shortest_paths(i))
		js = []
		for j in cited_authors: 
			j = a_2_node[j]
			js.append(j)
		cited_paths.append(yg.shortest_paths(i,js))
	cited_paths = np.array(cited_paths).reshape(1,-1)
	cited_paths[np.isinf(cited_paths)] = -1
	cited_paths[cited_paths==-1] = np.nanmax(cited_paths) + 1

	all_paths = np.array(all_paths).reshape(2,-1)
	all_paths[np.isinf(all_paths)] = -1
	all_paths[all_paths==-1] = np.nanmax(all_paths) + 1

	null_walks = 0

	base_matrix = np.zeros((8,8))
	base_matrix[:] = 0

	#if we take random walks, the length of the longest random path between two nodes, what is the base rate?
	# we only count times when the walk ends on a first or last author and paper was published before this paper
	
	if walk_length[:3] == 'all':
		try:
			multi = int(walk_length[4])
			wl = int(np.nanmax(all_paths))*multi
		except: wl = int(np.nanmax(all_paths))
		if wl == 0:
			base_matrix[:] = np.nan
			return [paper,base_matrix,np.mean(cited_paths),np.mean(all_paths)]
		for i in citing_authors:
			i = a_2_node[i.strip()]
			while True:
				walk_2 = i # start from this author
				for w in range(wl):
					walk_2 = np.random.choice(yg.neighbors(walk_2),1)[0]
				cited_author = node_2_a[walk_2]
				walk_papers = a_2_paper[cited_author]
				np.random.shuffle(walk_papers)
				for walk_paper in walk_papers:
					walk_paper_df = paper_df.iloc[walk_paper]
					if walk_paper_df.year>= year: continue
					#you found a paper, store it in matrix! 
					base_matrix = base_matrix + prs[walk_paper]
					null_walks += 1
				if null_walks > 1000:break
	if walk_length == 'cited':
		null_walks = 0.
		while True:
			wl = np.random.choice(cited_paths[0],1)[0]#pick a length from citations
			choices = np.where(all_paths[0]==wl)[0]
			if len(choices) == 0: continue
			walk_paper = np.random.choice(choices,1)[0] #pick a cited author of that length, first author
			cited_author = node_2_a[walk_paper]
			walk_papers = a_2_paper[cited_author]
			np.random.shuffle(walk_papers)
			for walk_paper in walk_papers:
				walk_paper_df = paper_df.iloc[walk_paper]
				if walk_paper_df.year>= year: continue
				#you found a paper, store it in matrix! 
				base_matrix = base_matrix + prs[walk_paper]
				null_walks += 1
			if null_walks > 1000:break
		null_walks = 0.
		while True:
			wl = np.random.choice(cited_paths[0],1)[0]#pick a length from citations
			choices = np.where(all_paths[1]==wl)[0]
			if len(choices) == 0: continue
			walk_paper = np.random.choice(choices,1)[0] #pick a cited author of that length, last author
			cited_author = node_2_a[walk_paper]
			walk_papers = a_2_paper[cited_author]
			np.random.shuffle(walk_papers)
			for walk_paper in walk_papers:
				walk_paper_df = paper_df.iloc[walk_paper]
				if walk_paper_df.year>= year: continue
				#you found a paper, store it in matrix! 
				base_matrix = base_matrix + prs[walk_paper]
				null_walks += 1
			if null_walks > 1000:break

	return [paper,base_matrix,np.mean(cited_paths),np.mean(all_paths)]

def make_year_graphs(year):
	global g
	print (year)
	yg = g.copy()
	yg.es.select(year_gt=year).delete()
	m = np.array(yg.components().membership)
	labels,counts = np.unique(m,return_counts=True)
	big = np.argmax(counts)
	return [year,[yg,m,big]]

def shortest_paths():
	global paper_df
	global main_df
	global graphs
	global node_2_a
	global a_2_node
	global a_2_paper
	global g
	global walk_length
	g = igraph.load('/%s/data/%s_coa_graph'%(homedir,method),format='pickle')
	paper_df = pd.read_csv('/%s/data/result_df_%s.csv'%(homedir,method))
	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)


	a_2_node = np.load('/%s/data/%s_a_2_node.npy'%(homedir,method),allow_pickle='TRUE').item()
	a_2_paper = np.load('/%s/data/%s_a_2_paper.npy'%(homedir,method),allow_pickle='TRUE').item()
	node_2_a = np.load('/%s/data/%s_node_2_a.npy'%(homedir,method),allow_pickle='TRUE').item()

	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	graphs = dict(pool.map(make_year_graphs,np.arange(2009,2020)))

	del pool
	paper_idxs = []
	for i,p in main_df.iterrows():
		year = p.PY
		if year >=2009:
			if type(p.CP) != np.float:
				if len(p['CP'].split(',')) >= 10:
					if graphs[year][1][a_2_node[p.AF.split(';')[0].strip()]] == graphs[year][2]:
						if graphs[year][1][a_2_node[p.AF.split(';')[-1].strip()]] == graphs[year][2]:
							paper_idxs.append(i)

	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	r = pool.map(multi_short,paper_idxs)
	walk_prs = np.zeros((main_df.shape[0],144))
	walk_prs[:] = np.nan
	walk_all = np.zeros((main_df.shape[0]))
	walk_all[:] = np.nan
	walk_cite = np.zeros((main_df.shape[0]))
	walk_cite[:] = np.nan
	for result in r:
		walk_prs[result[0]] = result[1].flatten()
		walk_cite[result[0]] = result[2]
		walk_all[result[0]] = result[3]

	np.save('/%s/data/walk_probabilities_%s_%s.npy'%(homedir,method,walk_length),walk_prs)	
	np.save('/%s/data/walk_all_%s_%s.npy'%(homedir,method,walk_length),walk_all)	
	np.save('/%s/data/walk_cite_%s_%s.npy'%(homedir,method,walk_length),walk_cite)	

def shortest_pr_paths():
	global paper_df
	global main_df
	global graphs
	global node_2_a
	global a_2_node
	global a_2_paper
	global g
	global walk_length
	global prs 
	prs = np.load('/%s/data/result_pr_df_%s.npy'%(homedir,method))
	g = igraph.load('/%s/data/%s_coa_graph'%(homedir,method),format='pickle')
	paper_df = pd.read_csv('/%s/data/result_df_%s.csv'%(homedir,method))
	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)


	a_2_node = np.load('/%s/data/%s_a_2_node.npy'%(homedir,method),allow_pickle='TRUE').item()
	a_2_paper = np.load('/%s/data/%s_a_2_paper.npy'%(homedir,method),allow_pickle='TRUE').item()
	node_2_a = np.load('/%s/data/%s_node_2_a.npy'%(homedir,method),allow_pickle='TRUE').item()

	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	graphs = dict(pool.map(make_year_graphs,np.arange(2009,2020)))

	del pool
	paper_idxs = []
	for i,p in main_df.iterrows():
		year = p.PY
		if year >=2009:
			if type(p.CP) != np.float:
				if len(p['CP'].split(',')) >= 10:
					if graphs[year][1][a_2_node[p.AF.split(';')[0].strip()]] == graphs[year][2]:
						if graphs[year][1][a_2_node[p.AF.split(';')[-1].strip()]] == graphs[year][2]:
							paper_idxs.append(i)

	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	r = pool.map(multi_pr_shortv2,paper_idxs[::-1])


	walk_prs = np.zeros((main_df.shape[0],64))
	walk_prs[:] = np.nan
	walk_all = np.zeros((main_df.shape[0]))
	walk_all[:] = np.nan
	walk_cite = np.zeros((main_df.shape[0]))
	walk_cite[:] = np.nan
	for result in r:
		walk_prs[result[0]] = result[1].flatten()
		walk_cite[result[0]] = result[2]
		walk_all[result[0]] = result[3]

	np.save('/%s/data/walk_pr_probabilities_%s_%s.npy'%(homedir,method,walk_length),walk_prs)	
	np.save('/%s/data/walk_pr_all_%s_%s.npy'%(homedir,method,walk_length),walk_all)	
	np.save('/%s/data/walk_pr_cite_%s_%s.npy'%(homedir,method,walk_length),walk_cite)	

def cite_paths():

	main_df = pd.read_csv('/%s/article_data/NewArticleData2019.csv'%(homedir),header=0)
	race_df = pd.read_csv('/%s/data/result_df_%s.csv'%(homedir,method))
	df = race_df.merge(main_df,how='outer',left_index=True, right_index=True)

	walk_all = np.load('/%s/data/walk_all_%s_%s.npy'%(homedir,method,walk_length))	
	walk_cite = np.load('/%s/data/walk_cite_%s_%s.npy'%(homedir,method,walk_length))	

	mask = np.isnan(walk_cite)==False
	print (scipy.stats.ttest_rel(walk_all[mask],walk_cite[mask]))

	matrix = np.load('/%s/data/citation_matrix_pr_%s_%s.npy'%(homedir,method,control))
	base_matrix = np.load('/%s/data/base_citation_matrix_pr_%s_%s.npy'%(homedir,method,control))

	matrix_idxs = {'white_M':0,'api_M':1,'hispanic_M':2,'black_M':3,'white_W':4,'api_W':5,'hispanic_W':6,'black_W':7}

	if within_poc == False:
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W',]),
		np.vectorize(matrix_idxs.get)(['api_M','api_W','hispanic_M','hispanic_W','black_M','black_W',])]
		names = ['white-white','white-poc','poc-white','poc-poc']

	if within_poc == 'black':
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W',]),
		np.vectorize(matrix_idxs.get)(['black_M','black_W',])]
		names = ['nb-nb','nb-black','black-nb','black-black']

	if within_poc == 'api':
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W',]),
		np.vectorize(matrix_idxs.get)(['api_M','api_W',])]
		names = ['na-na','na-asian','asian-na','asian-asian']

	if within_poc == 'hispanic':
		groups = [np.vectorize(matrix_idxs.get)(['white_M','white_W',]),
		np.vectorize(matrix_idxs.get)(['hispanic_M','hispanic_W',])]
		names = ['nh-nh','nh-hispanic','hispanic-nh','hispanic-hispanic']


	plot_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))
	plot_base_matrix = np.zeros((matrix.shape[0],len(groups),len(groups)))

	for i in range(len(groups)):
		for j in range(len(groups)):
			plot_matrix[:,i,j] = np.nansum(matrix[:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)
			plot_base_matrix[:,i,j] = np.nansum(base_matrix[:,groups[i]][:,:,groups[j]].reshape(matrix.shape[0],-1),axis=1)


	n_iters = 10000

	white_data = pd.DataFrame(columns=['citation_rate','citation_type','path_length'])
	papers = df[(df.year>2009)&((df.fa_race=='white')&(df.la_race=='white'))]
	for boot in range(n_iters):
		boot_papers = papers.sample(100,replace=False).index
		
		emperical = np.nanmean(plot_matrix[boot_papers],axis=0)
		expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
		
		emperical = emperical / np.sum(emperical)
		expected = expected / np.sum(expected)

		rate = (emperical - expected) / expected
		
		pl = np.nanmean(walk_cite[boot_papers])
		# np.sum(rate.flatten()[1:])-rate[0,0]
		
		tdf = pd.DataFrame(data=np.array([rate.flatten(),names]).swapaxes(0,1),columns=['citation_rate','citation_type'])
		tdf['path_length'] = pl
		white_data = white_data.append(tdf,ignore_index=True)   
	
	white_data.citation_rate = (white_data.citation_rate.astype(float)*100) #are you citing aoc well?

		
	for perc in [10,15,20,25]:

		x = []
		y = []
		for name in names[3:]:
			cut_off = np.percentile(white_data[(white_data.citation_type==name)].citation_rate,perc)
			x.append(white_data[(white_data.citation_type==name)&(white_data.citation_rate>cut_off)].path_length.values)
			y.append(white_data[(white_data.citation_type==name)&(white_data.citation_rate<cut_off)].path_length.values)
		x = np.array(x).flatten()
		y = np.array(y).flatten()
		print (scipy.stats.ttest_ind(x,y))
		
	for perc in [10,15,20,25]:
		x = []
		y = []
		cut_off = np.percentile(white_data[(white_data.citation_type=='white-white')].citation_rate,perc)
		x.append(white_data[(white_data.citation_type=='white-white')&(white_data.citation_rate>cut_off)].path_length.values)
		y.append(white_data[(white_data.citation_type=='white-white')&(white_data.citation_rate<cut_off)].path_length.values)
		x = np.array(x).flatten()
		y = np.array(y).flatten()
		print (scipy.stats.ttest_ind(x,y))

	data = pd.DataFrame(columns=['citation_rate','path_length'])
	papers = df[(df.year>2009)&((df.fa_race!='white')|(df.la_race!='white'))]
	for boot in range(n_iters):
		boot_papers = papers.sample(100,replace=False).index
		
		emperical = np.nanmean(plot_matrix[boot_papers],axis=0)
		expected = np.nanmean(plot_base_matrix[boot_papers],axis=0)
		
		emperical = emperical / np.sum(emperical)
		expected = expected / np.sum(expected)

		rate = (emperical - expected) / expected
		
		pl = np.nanmean(walk_cite[boot_papers])

		tdf = pd.DataFrame(data=np.array([rate.flatten(),names]).swapaxes(0,1),columns=['citation_rate','citation_type'])
		tdf['path_length'] = pl
		data = data.append(tdf,ignore_index=True)  

	data.citation_rate = (data.citation_rate.astype(float)*100) #are you citing aoc well?

	for perc in [10,15,20,25]:


		x = []
		y= []
		for name in names[1:]:
			cut_off = np.percentile(data[(data.citation_type==name)].citation_rate,perc)
			x.append(data[(data.citation_type==name)&(data.citation_rate>cut_off)].path_length.values)
			y.append(data[(data.citation_type==name)&(data.citation_rate<cut_off)].path_length.values)
		x = np.array(x).flatten()
		y = np.array(y).flatten()
		print (scipy.stats.ttest_ind(x,y))


	for perc in [10,15,20,25]:
		x = []
		y= []
		cut_off = np.percentile(data[(data.citation_type=='white-white')].citation_rate,perc)
		x.append(data[(data.citation_type=='white-white')&(data.citation_rate>cut_off)].path_length.values)
		y.append(data[(data.citation_type=='white-white')&(data.citation_rate<cut_off)].path_length.values)
		x = np.array(x).flatten()
		y = np.array(y).flatten()
		print (scipy.stats.ttest_ind(x,y))

"""
ORDER 

run 'make_df'
run 'make_all_author_race'
run 'make_all_percentages'
run 'make_networks'

"""


