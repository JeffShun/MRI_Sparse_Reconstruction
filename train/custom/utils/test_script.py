import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
import custom  # noqa: F401

version = "v1"
test_epochs = range(50,100,2)
out_dir = "../example/data/%s_output"%(version)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
log_path = out_dir + "/test.log"

for i in test_epochs:
    f = open(log_path, 'a+')  
    print("Begin test epoch %d !"%(i), file=f)
    f.close()
    os.chdir("../example")
    os.system("python main.py --output_path ./data/%s_output/seg --model_file ../train/checkpoints/%s/epoch_%d.pth"%(version,version,i))
    os.chdir("../train")
    os.system("python custom/utils/cal_matrics.py --pred_path ../example/data/%s_output/seg --output_path ../example/data/%s_output/logs/%d.csv --print_path %s"%(version,version,i,log_path))

