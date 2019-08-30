#Python code to run a simple Word2Vec NLP example
#glove embeddings https://nlp.stanford.edu/projects/glove/
#questions-words.txt download https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt
#gensim api reference for Word2Vec: https://radimrehurek.com/gensim/models/word2vec.html

import gensim
import gensim.downloader as api
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse

#A function to generate a single sentence from a file
def genSentences(inFile, num):
  with open(inFile, 'r') as f:
    for i,line in enumerate(f):
      if num>0 and i>num: break
      seq = line.strip()
      if sent: yield gensim.utils.simple_preprocess(sent)

#An iterator to generate a single sentence at a time from the given file
class SentenceIter:
  def __init__(self, inFile, num=-1):
    self.inFile = inFile
  def __iter__(self):
    return genSentences(self.inFile, self.num)

#Create a Word2Vec embedding based in the given set of sentences
def embedText(inFile, numSentences=-1, dim=30):
  sentences = SentenceIter(inFile, num=numSentences)
  model = models.Word2Vec(sentences, #iterator producing a list of lists where each sublist is a sentence
                          size=dim, #dimension of the final embedding, typically 100-300
                          window=10, #max distance from each input word to consider
                          min_count=5, #minimum number of times a word must occur to be considered
                          sg=1, #1 for skipgram, cbow otherwise
                          hs=0, #1 for hierarchical softmax, 0 for negative sampling
                          negative=10, #number of negative examples per positive example
                                       #2-5 for large datasets, 5-20 for smaller sets #FROM WHERE?
                          iter=3, #number of training epochs
                          workers=1) #number of threads to use
  return model

#Convert a Glove model to a Word2Vec model
def gloveFromW2V(gloveFile, save=''):
  w2vFile = '/'.join(gloveFile.split('/')[:-1]) + 'w2v_'+gloveFile.split('/')[-1]
  glove2word2vec(gloveFile, w2vFile if not save else save)
  return loadW2V(w2vFile)

#Load a Word2Vec model
def loadW2V(w2vFile):
  return gensim.models.KeyedVectors.load_word2vec_format(w2vFile, binary=True if ('bin' in w2vFile.split('.')) else False)

#Print top n similarities related to the king/queen analogy
def kingQueenTest(model, n=5):
  nearKing = model.most_similar(positive='king', topn=n)
  nearMan = model.most_similar(positive='man', topn=n)
  nearQueen = model.most_similar(positive='queen', topn=n)
  nearWoman = model.most_similar(positive='woman', topn=n)
  king_queen = model.n_similarity(['king'], ['queen'])
  king_minus_man = model.most_similar(positive=['king'], negative=['man'], topn=n)
  queen_minus_woman = model.most_similar(positive=['queen'], negative=['woman'], topn=n)
  king_to_queen = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=n)
  queen_to_king = model.most_similar(positive=['queen', 'man'], negative=['woman'], topn=n)

  print('********************************')
  print('Most similar to \'king\'')
  for i,w in enumerate(nearKing): print('   '+str(i)+': '+w[0])
  print('Most similar to \'queen\'')
  for i,w in enumerate(nearQueen): print('   '+str(i)+': '+w[0])
  print('Most similar to \'man\'')
  for i,w in enumerate(nearMan): print('   '+str(i)+': '+w[0])
  print('Most similar to \'woman\'')
  for i,w in enumerate(nearWoman): print('   '+str(i)+': '+w[0])
  print('Similarity of \'king\' and \'queen\': '+str(king_queen))
  print('Most similar to \'king\' - \'man\'')
  for i,w in enumerate(king_minus_man): print('   '+str(i)+': '+w[0])
  print('Most similar to \'queen\' - \'woman\'')
  for i,w in enumerate(queen_minus_woman): print('   '+str(i)+': '+w[0])
  print('Most similar to \'king\' - \'man\' + \'woman\'')
  for i,w in enumerate(king_to_queen): print('   '+str(i)+': '+w[0])
  print('Most similar to \'queen\' - \'woman\' + \'man\'')
  for i,w in enumerate(queen_to_king): print('   '+str(i)+': '+w[0])
  print('********************************')

#Run the analogy test for a given model and set of questions
#By default the testFile is questions-words.txt
def analogyTest(model, testFile='questions-words.txt'):
  score, results = model.evaluate_word_analogies(testFile)
  print(list(results[0].keys()))
  for section in results:
    print(section['section'], len(section['correct']), len(section['incorrect']))
  print('Analogy score: '+str(score))

#Given a model and a list of words, plot the word embeddings in two dimensions using TSNE
def plotWords(model, wordList):
  locs = [model[w] for w in wordList]

  tsne = TSNE(n_components=2, init='pca')
  (X, Y) = zip(*tsne.fit_transform(locs))

  plt.scatter(X, Y, c='b', s=5, alpha=0.5)
  for (label,x,y) in zip(wordList, X, Y):
    plt.annotate(label, xy=(x,y), xytext=(0,0), textcoords='offset points')
  plt.show()

#Collect to n nearest words to a given word according to a model
def nearbyWords(model, word, n=10):
  return [elem[0] for elem in model.most_similar(positive=word, topn=n)]

#Collect n random words in a model
def randomWords(model, n=10):
  return list(np.random.choice(list(model.vocab.keys()), size=n, replace=False))

#Run some example tests
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Provide parameters to run simple examples using Word2Vec')
  parser.add_argument('--sentences', type=str, default='', required=False,
                      help='A file containing a single sentence per line')
  parser.add_argument('--numSentences', type=int, default=-1, required=False,
                      help='The maximum number of sentences to use in training in the model (-1, the default, means use all sentences)')
  parser.add_argument('--dim', type=int, default=100, required=False,
                      help='The embedding dimension')
  parser.add_argument('--w2v', type=str, default='', required=False,
                      help='A Word2Vec model to use. Defaults to ...')
  parser.add_argument('--glove', type=str, default='', required=False,
                      help='A Glove model to convert to Word2Vec and use')
  parser.add_argument('--analogyFile', type=str, default='', required=False,
                      help='A file with analogy questions for testing a model')
  parser.add_argument('--kingQueenTest', type=int, default=0, required=False,
                      help='Run the king/queen analogy test and print this many nearest vectors')
  parser.add_argument('--word', type=str, default='', required=False,
                      help='A word to plot along with nearby and random words')
  parser.add_argument('--randWord', action='store_true',
                      help='If set, plot a random word along with nearby and other random words')
  parser.add_argument('--nearby', type=int, default=10, required=False,
                      help='A number of nearby words to plot')
  parser.add_argument('--random', type=int, default=10, required=False,
                      help='A number of random words to plot')
  args = parser.parse_args()

  model = None
  if args.sentences: model = embedText(args.sentences, numSentences=args.numSentences, dim=args.dim)
  elif args.w2v: model = loadW2V(args.w2v)
  elif args.glove: model = gloveFromW2V(args.glove, save='')
  else: model = loadW2V('w2v_glove.6B.100d.txt')

  if args.analogyFile: analogyTest(model, testFile=args.analogyFile)
  if args.kingQueenTest>0: kingQueenTest(model, n=args.kingQueenTest)
  if args.word or args.randWord:
    word = args.word if args.word else np.random.choice(list(model.vocab.keys()))
    print('Word: '+word)
    nearby = nearbyWords(model, word, n=args.nearby)
    print('Nearby words: '+', '.join(nearby))
    random = randomWords(model, n=args.random)
    print('Random words: '+', '.join(random))
    plotWords(model, [word]+nearby+random)

  



