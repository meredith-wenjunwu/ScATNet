import glob
import os
import json

runs_dir = '/projects/patho4/Kechun/diagnosis/melanocyte_attention/model/1scale/10x/49/sma_guide_frobenius'
results = glob.glob(os.path.join(runs_dir, '*', 'test_case_summary.json'))
for result in results:
    stats = json.load(open(result))
    print(result.split('/')[-2], stats['overall_accuracy'])
