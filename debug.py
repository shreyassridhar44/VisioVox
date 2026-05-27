
import sys
sys.path.insert(0, '.')
from preprocess_devA import process_clip, OUTPUT_DIR

test_clip = r'E:\vox2_dev_mp4_partaa~\dev\mp4\id00019\_lmvY4AiroM\00121.mp4'
args = (test_clip, 'id00019_test_00121', 'id00019', OUTPUT_DIR)
print('Testing single clip end to end...')
record, error = process_clip(args)
print('Record:', record)
print('Error:', error)
