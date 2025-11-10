import sys
import getopt
from vectorize import img2vec_converter
from cos_sim import matrix_calculator
from top_similar import top10
from pathlib import Path
import os
import pickle


def usage():
    print("Usage: python similarity_matrix.py [-c cuda] [-m model] [-f fine_tuned_path] [-i input_dir]")
    print("  -c --cuda          True/False (default False)")
    print("  -m --model         model name (default resnet18)")
    print("  -f --fine_tuned    path to fine-tuned model state_dict (optional)")
    print("  -i --input_dir     directory with resized images (default: resized)")


cuda = False
model = "resnet18"
fine_tuned_path = None
input_dir = 'resized'

try:
    opts, args = getopt.getopt(sys.argv[1:], "c:m:f:i:h", ['cuda=', 'model=', 'fine_tuned=', 'input_dir=', 'help'])
except getopt.GetoptError:
    usage()
    sys.exit(2)

if not opts:
    print("Using default options. Model: resnet18, Cuda: False, input_dir: resized")
else:
    for opt, arg in opts:
        if opt in ('-c', '--cuda'):
            cuda = arg in ('True', 'true', '1')
        elif opt in ('-m', '--model'):
            model = arg
        elif opt in ('-f', '--fine_tuned'):
            fine_tuned_path = arg
        elif opt in ('-i', '--input_dir'):
            input_dir = arg
        elif opt in ('-h', '--help'):
            usage()
            sys.exit()
        else:
            print("Check your arguments")
            sys.exit(2)

print(f"Converting images from '{input_dir}' using model={model}, cuda={cuda}, fine_tuned={fine_tuned_path}")

vectors = img2vec_converter(model, cuda, fine_tuned_path, input_dir=input_dir)
matrix = matrix_calculator(vectors)

print("Saving cosine similarity matrix")
os.makedirs(os.path.join("pickles", model), exist_ok=True)
path_output = os.path.join('pickles', model, 'sim_matrix.pkl')
with open(path_output, 'wb') as f:
    pickle.dump(matrix, f)

top10(matrix, model)