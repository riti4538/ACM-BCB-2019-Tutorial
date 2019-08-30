#Python code to make a file from human_cds.txt with one sequence per line

import argparse

def makeSequenceFile(inFile, outFile):
  with open(inFile, 'r') as f, open(outFile, 'w') as o:
    seq = ''
    f.readline() #first > line
    for line in f:
      if line[0]!='>' and all(map(lambda c: c in ['A', 'C', 'T', 'G'], line.strip())): seq += line.strip()
      elif len(seq)>0:
        o.write(seq+'\n')
        seq = ''

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Provide in and out files to make a sequence file')
  parser.add_argument('--i', type=str, default='human_cds.txt', required=False,
                      help='in file name')
  parser.add_argument('--o', type=str, default='../sequences.txt', required=False,
                      help='out file name')
  args = parser.parse_args()

  makeSequenceFile(args.i, args.o)

