import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import EnvioX
import ReNet

model_mode = 1

if model_mode == 1:
    model = ReNet.ReNet1(in_channels=3,
                         out_channels=3,
                         D=4,
                         alpha=0.5
                         )
elif model_mode == 2:
    model = ReNet.ReNet2(in_channels=3,
                         out_channels=3,
                         D=4,
                         alpha=0.5
                         )
    
model_path = "best_ReNet.pth"
test_data_path = "data/test"
data_index = 2
def main():
    EnvioX.Visualizer.visualize_model_result(model=model,
                                             model_path=model_path,
                                             data_path=test_data_path,
                                             data_index=data_index
                                             )
    
if __name__ == "__main__":
    main()