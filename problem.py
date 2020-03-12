from tensor2tensor import problems

PROBLEM = 'translate_enfr_wmt32k'
TMP_DIR = '/tmp' # Where data files from internet stored
DATA_DIR = '/data' # Where pre-prcessed data is stored

# Init problem T2T object the generated training data
t2t_problem = problems.problem(PROBLEM)
t2t_problem.generate_data(DATA_DIR, TMP_DIR) 