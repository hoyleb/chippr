import os, re
#import subprocess
#import argparse
def clean(fileName):
	remove_keys = ['abstract', 'annote', 'isbn', 'mendeley-groups', 'keywords', 'file', 'issn']
	blind_keys = ['url', 'language']
	bibf = open(fileName, 'r')
	cleanf = open(fileName + '.tmp', 'w')
	lines = bibf.readlines()
	for line in lines:
		if re.search('^'+'|'.join(blind_keys),line):
			cleanf.write(line)
		elif not re.search('^'+'|'.join(remove_keys),line):
			#print line
			cleanf.write(line)
	cleanf.close()
	os.remove(fileName)
	os.rename(fileName + '.tmp', fileName)
def readable(infile, outfile):
	intfile = 'temp.tex'
	os.system('sed \'/^%/ d\' < ' + infile + ' > ' + intfile)
	os.system('fold -w80 -s ' + intfile + ' > ' + outfile)

clean('references.bib')

cwd = os.getcwd()
infile = os.path.join(cwd, 'draft-editing.tex')
outfile = os.path.join(cwd, 'draft.tex')
print(infile, outfile)
readable(infile, outfile)

