#Python code using multilateration to generate embeddings of Hamming graphs and modified similarity matrices for codons

import numpy as np
from scipy.special import comb
from itertools import combinations, product
from multilateration import checkResolving, checkResolvingHamming, hammingDist, ich
import argparse

#Determine the metric dimension of a Hamming graph via brute force search starting with a given set size
def metricDimension(k, alphabet, size=1, verbose=False):
  while True:
    if verbose: print('size', size)
    for i,R in enumerate(combinations(product(alphabet, repeat=k), size)):
      if verbose and i%1000==0: print('   ', 'k', k, 'a', len(alphabet), 'size', size, i)
      if checkResolvingHamming(R, k, alphabet): return size
    size += 1

#Find up to maxNum random resolving sets of a Hamming graph
def resolvingSets(k, alphabet, size=-1, maxNum=1, verbose=False):
  if size==-1: size = metricDimension(k, size=1)
  resSets = {}
  total = max(comb(np.power(len(alphabet), k), size), 1000)
  i = 0
  while len(resSets)<maxNum and i<total:
    if verbose and i%1000==0: print('num res sets found: ', len(resSets), 'k', k, 'a', len(alphabet), 'i', i)
    i += 1
    R = randomCombo(k, alphabet, size)
    if checkResolvingHamming(R, k, alphabet):
      resSets[R] = 1
      if len(resSets)>=maxNum: return list(resSets)
  return list(resSets)

#Generate a random set of kmers
def randomCombo(k, alphabet, size):
  kmers = product(alphabet, repeat=k)
  pool = tuple(kmers)
  n = len(pool)
  indices = sorted(np.random.choice(range(n), size=size, replace=False))
  return tuple(pool[i] for i in indices)

#Determine the multilateration based embedding of all kmers based on a given set
def embedding(k, alphabet, R):
  locs = {} #{kmer: [hammingDist(kmer, r) for r in R] for kmer in product(alphabet, repeat=k)}
  for kmer in product(alphabet, repeat=k):
    locs[''.join(''.join(kmer))] = [hammingDist(kmer, r) for r in R]
  return locs

#Save a set of resolving sets to a file
def saveSets(RList, outFile):
  with open(outFile, 'w') as o:
    for R in RList:
      o.write('\t'.join([''.join(r) for r in R])+'\n')

#Save a multilateration based embedding of a Hamming graph based on a given set
def saveEmbedding(locs, k, a, R, outFile):
  with open(outFile, 'w') as o:
    o.write('Embedding of H_{'+str(k)+','+str(a)+'} with R='+','.join(''.join(r) for r in R)+'\n')
    for kmer in sorted(locs.keys()):
      o.write('\t'.join([kmer]+list(map(str, locs[kmer])))+'\n')

#Generate a similarity matrix for codons given a list of weights
#weights give a value to multiply the Hamming distance by for each index
#e.g. for weights = [3,2,1], d(ATC, TGG) = 3*1+2*1+1*1 = 6
#     d(ATC, ACC) = 3*0+2*1+1*0 = 2
def codonSimMatrix(weights):
  if len(weights)<3: weights += [1 for _ in range(3-len(weights))]
  dist = lambda a,b: sum(w for (x,y,w) in zip(a,b,weights) if x!=y)
  codons = list(product(['A', 'C', 'G', 'T'], repeat=3))
  M = [[dist(A,B) for B in codons] for A in codons]
  return M

#Find up to maxNum random resolving sets of a codon similarity matrix
def codonResolvingSets(M, size=-1, maxNum=1, verbose=False):
  resSets = {}
  codons = list(product(['A', 'C', 'G', 'T'], repeat=3))
  total = max(comb(64, size), 1000)
  i = 0
  while len(resSets)<maxNum and i<total:
    if verbose and i%1000==0: print('num res sets found: ', len(resSets), 'i', i)
    i += 1
    R = randomCombo(3, ['A', 'C', 'G', 'T'], size)
    Ri = [codons.index(r) for r in R]
    if checkResolving(Ri, M):
      resSets[R] = 1
      if len(resSets)>=maxNum: return list(resSets)
  return list(resSets)

#Determine the multilateration based embedding of codons based on a given set and similarity matrix
def codonEmbedding(M, R):
  locs = {}
  codons = list(product(['A', 'C', 'G', 'T'], repeat=3))
  for codon in codons:
    locs[codon] = [M[codons.index(r)][codons.index(codon)] for r in R]
  return locs

#Save a multilateration based embedding of a codon similarity matrix based on a given set
def saveCodonEmbedding(locs, weights, R, outFile):
  with open(outFile, 'w') as o:
    o.write('Codon Embedding with weights '+','.join([str(w) for w in weights])+' and resolving set '+','.join([''.join(r) for r in R])+'\n')
    for codon in locs:
      o.write('\t'.join([''.join(codon)]+[str(d) for d in locs[codon]])+'\n')

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Generate Embeddings for H_{k,4}')
  parser.add_argument('--kList', nargs='*', type=int, default=[3], required=False,
                      help='a list of string lengths to consider') #1,2,3,4,5
  parser.add_argument('--betas', nargs='*', type=int, default=[], required=False,
                      help='if nonempty and the same length as kList, these are assumed to be the metric dimension of H_{k,4} for each k in kList') #3,4,6,8,10 #last two from upper bound on k=3
  parser.add_argument('--weights', nargs='*', type=float, default=[], required=False,
                      help='if nonempty find resolving sets using a codon similarity matrix generated with the given weights (only the first three are used)')
  parser.add_argument('--numResSets', type=int, default=50, required=False,
                      help='the number of random resolving sets to save for each k')
  parser.add_argument('--numEmbeddings', type=int, default=1, required=False,
                      help='the number of embeddings to save for each k')
  args = parser.parse_args()


  alphabet = ['A', 'C', 'T', 'G']
  kList = sorted(args.kList)
  betas = {k:-1 for k in kList}

  #Determine the metric dimension of Hamming graphs with given list of sequence lengths (k values)
  if len(args.betas)!=len(kList):
    size = 1
    for k in kList:
      betas[k] = metricDimension(k, alphabet, size=size, verbose=True)
      size = betas[k]
  else: betas = {k:args.betas[i] for i,k in enumerate(args.kList)}

  #Find random resolving sets and save associated multilateration based embeddings
  for k in kList:
    RList = resolvingSets(k, alphabet, size=betas[k], maxNum=args.numResSets, verbose=True)
    saveSets(RList, 'resolving_sets_H_'+str(k)+'_4.tsv')
    for i,index in enumerate(np.random.choice(len(RList), size=min(len(RList), args.numEmbeddings), replace=False)):
      locs = embedding(k, alphabet, RList[index])
      saveEmbedding(locs, k, len(alphabet), RList[index], 'H_'+str(k)+'_4_embedding_'+str(i)+'.tsv')

  #if weights given, find and save resolving sets for a codon similarity matrix
  if len(args.weights)>0:
    M = codonSimMatrix(args.weights[:3])
    beta = len(ich(M, randOrder=True))
    RList = codonResolvingSets(M, size=beta, maxNum=args.numResSets, verbose=True)
    saveSets([[''.join(r) for r in R] for R in RList], 'resolving_sets_codons_'+','.join([str(w) for w in args.weights])+'.tsv')
    for i,index in enumerate(np.random.choice(len(RList), size=min(len(RList), args.numEmbeddings), replace=False)):
      locs = codonEmbedding(M, RList[index])
      saveCodonEmbedding(locs, args.weights, RList[index], 'codon_embedding_'+','.join([str(w) for w in args.weights])+'_'+str(i)+'.tsv')


