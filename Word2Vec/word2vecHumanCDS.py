#Python code to generate Word2Vec embeddings of DNA k-mers based on a given set of sequences
#In this case these sequences are human coding sequences

from gensim import models
import logging
import argparse

#A function to generate a single sequence from the file
#Up to k reading frames are considered
def genSentences(inFile, k, frames=1):
  frames = min(frames, k)
  with open(inFile, 'r') as f:
    for line in f:
      seq = line.strip()
      for i in range(frames):
        #Split seq into chunks of k characters starting at the ith position
        sent = list(map(''.join, zip(*[iter(seq[i:])]*k)))
        if sent: yield sent
        else: break

#An iterator to generate a single sentence at a time from the given file
class SentenceIter:
  def __init__(self, inFile, k, frames):
    self.inFile = inFile
    self.k = k
    self.frames = frames
  def __iter__(self):
    return genSentences(self.inFile, self.k, frames=self.frames)

#Create a Word2Vec embedding based in the given set of sentences
def embedKmers(inFile, k, dim=30, frames=1):
  sentences = SentenceIter(inFile, k, frames=frames)
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

#Generate a Word2Vec embedding of kmers and save the result
def saveEmbeddings(inFile, kList, outFileList=[], dimList=[30], frames=[1], saveModel=False):
  for i,k in enumerate(kList):
    model = embedKmers(inFile, k, dim=dimList[min(i,len(dimList)-1)], frames=frames[i])
    outFile = outFileList[i] if len(outFileList)>i else 'w2v_embedding_'+str(k)+'mers.locs'
    with open(outFile, 'w') as o:
      o.write('H_'+str(k)+',4 word2vec embedding with '+str(frames[i])+'frames\n') #header
      for word in sorted(model.wv.vocab):
        o.write('\t'.join([word]+list(map(str,model[word])))+'\n') ###NOT using pickle since want to see in file and not huge spaces... typically might want to compress somehow
    if saveModel: model.save('w2v_model_'+str(k)+'mers.model') #saves full Word2Vec model

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Provide in and out files to make a sequence file')
  parser.add_argument('--i', type=str, default='../Data/sequences.txt', required=False,
                      help='name of file containing sentences to provide to Word2Vec')
  parser.add_argument('--kList', nargs='*', type=int, default=[3], required=False,
                      help='a list of kmer sizes') #1,2,3,4,5
  parser.add_argument('--o', nargs='*', type=str, default=[], required=False,
                      help='a list of files where embedding information will be saved')
  parser.add_argument('--dimList', nargs='*', type=int, default=[30], required=False,
                      help='list of embedding dimensions to use') #3,4,6,8,10 for ks above
  parser.add_argument('--frames', nargs='*', type=int, default=[1], required=False,
                      help='the maximum number of reading frames to consider for each sequence')
  parser.add_argument('--saveModel', action='store_true',
                      help='flag to save full word2vec model along with embedding')
  parser.add_argument('--log', action='store_true',
                      help='flag to enable logging in Word2Vec')
  args = parser.parse_args()

  if args.log: logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

  if len(args.frames)<len(args.kList): args.frames += [args.frames[-1] for _ in range(len(args.kList)-len(args.frames))]
  saveEmbeddings(args.i, args.kList, outFileList=args.o, dimList=args.dimList, frames=args.frames, saveModel=args.saveModel)

#  python3 word2vecHumanCDS.py --kList 1 2 3 4 5 --dimList 3 4 6 8 10 --frames 1 --log
#  python3 word2vecHumanCDS.py --kList 1 2 3 4 5 --dimList 3 4 6 8 10 --frames 1 2 3 4 5 --o multiframe_w2v_embedding_1mers.locs multiframe_w2v_embedding_2mers.locs multiframe_w2v_embedding_3mers.locs multiframe_w2v_embedding_4mers.locs multiframe_w2v_embedding_5mers.locs --log
#  python3 word2vecHumanCDS.py --kList 1 2 3 4 5 --dimList 4 16 64 256 1024 --frames 1 --o max_dim_w2v_embedding_1mers.locs max_dim_w2v_embedding_2mers.locs max_dim_w2v_embedding_3mers.locs max_dim_w2v_embedding_4mers.locs max_dim_w2v_embedding_5mers.locs --log
#  python3 word2vecHumanCDS.py --kList 1 2 3 4 5 --dimList 4 16 64 256 1024 --frames 1 2 3 4 5 --o multiframe_max_dim_w2v_embedding_1mers.locs multiframe_max_dim_w2v_embedding_2mers.locs multiframe_max_dim_w2v_embedding_3mers.locs multiframe_max_dim_w2v_embedding_4mers.locs multiframe_max_dim_w2v_embedding_5mers.locs --log

