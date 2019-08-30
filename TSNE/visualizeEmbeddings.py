#Python code using TSNE to visualize embeddings

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import glob
import argparse

#Read an embedding
def readEmbedding(inFile):
  locs = {}
  with open(inFile, 'r') as f:
    f.readline()
    for line in f:
      l = line.strip().split('\t')
      locs[l[0]] = list(map(float, l[1:]))
  return locs

#Use TSNE to generate a simple visualization of a given embedding
#groups is a dictionary of label -> [kmers]
def visualize(locs, groups, textLabels={}, legend=True):
  #Use TSNE to generate a two-dimensional version of the embedding
  kmers = sorted(list(locs))
  tsne = TSNE(n_components=2, init='pca')
  (X, Y) = zip(*tsne.fit_transform([locs[kmer] for kmer in kmers]))

  colors = ['b', 'g', 'r', 'c', 'm', 'k']
  markers = ['o', '^', '+', 'x', '*']
  symb = [(c,m) for m in markers for c in colors]

  #Plot kmers in the same group together with a common symbol/color
  for i,label in enumerate(groups):
    groupX = [X[kmers.index(kmer)] for kmer in groups[label]]
    groupY = [Y[kmers.index(kmer)] for kmer in groups[label]]
    plt.scatter(groupX, groupY, s=25, c=symb[i][0], marker=symb[i][1], label=label+'('+str(len(groups[label]))+')', alpha=0.5)

  #Add any text labels
  for kmer in textLabels:
    plt.annotate(textLabels[kmer], xy=(X[kmers.index(kmer)], Y[kmers.index(kmer)]), xytext=(0,0), textcoords='offset points')

  ax = plt.gca()
  ax.get_xaxis().set_ticks([])
  ax.get_yaxis().set_ticks([])
  plt.legend(loc='lower right')
  plt.show()

#Read a set of groups from a given file
#Only the proints provided are assigned groups
def readGroups(groupFile, points):
  if len(glob.glob(groupFile))==0: return {'': points}
  groups = {}
  with open(groupFile, 'r') as f:
    for line in f:
      l = line.strip().split('\t')
      for kmer in l[1:]:
        if kmer in points: del points[kmer]
      groups[l[0]] = l[1:]
  if len(points)>0: groups[''] = list(points)#.keys()
  return groups

#Read text labels for specific kmers from a file
def readTextLabels(inFile):
  if len(glob.glob(inFile))==0: return {}
  labels = {}
  with open(inFile, 'r') as f:
    for line in f:
      l = line.strip().split('\t')
      labels[l[0]] = l[1]
  return labels

#Create a file grouping codons by associated ammino acid
def codonGroups(outFile):
  codonMap = {'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T', 'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R', 'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P', 'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R', 'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G', 'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L', 'TAC':'Y', 'TAT':'Y', 'TAA':'STOP', 'TAG':'STOP', 'TGC':'C', 'TGT':'C', 'TGA':'STOP', 'TGG':'W'}
  aaMap = {}
  for kmer in codonMap:
    aa = codonMap[kmer]
    if aa not in aaMap: aaMap[aa] = []
    aaMap[aa].append(kmer)
  with open(outFile, 'w') as o:
    for aa in sorted(aaMap.keys()):
      o.write('\t'.join([aa]+aaMap[aa])+'\n')


if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Visualize an embedding using TNSE')
  parser.add_argument('--i', type=str, default='../Multilateration/H_3_4_embedding_0.tsv', required=False,
                      help='embedding file to be visualized')
  parser.add_argument('--groupsFile', type=str, default='', required=False,
                      help='file containing information about groups of points to label')
  parser.add_argument('--textFile', type=str, default='', required=False,
                      help='file containing infomation about points to label')
  args = parser.parse_args()

  locs = readEmbedding(args.i)

  groups = readGroups(args.groupsFile, {k:1 for k in locs})
  textLabels = readTextLabels(args.textFile)

  visualize(locs, groups, textLabels=textLabels, legend=True)

