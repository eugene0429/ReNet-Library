import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from ReNet.model1 import ReNet as ReNet1 # type: ignore
from ReNet.model2 import ReNet as ReNet2 # type: ignore
from EnvioX.Visualizer import Visualizer # type: ignore

model_mode = 1

if model_mode == 1:
    model = ReNet1(in_channels=3,
                   out_channels=3,
                   D=4,
                   alpha=0.5
                   )
elif model_mode == 2:
    model = ReNet2(in_channels=3,
                   out_channels=3,
                   D=4,
                   alpha=0.5
                   )
    
model_path = "best_ReNet.pth"
test_data_path = "data/test"
data_index = 2
def main():
    Visualizer.visualize_model_result(model=model,
                                      model_path=model_path,
                                      data_path=test_data_path,
                                      data_index=data_index
                                      )
    
if __name__ == "__main__":
    main()