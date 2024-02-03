import torch
from model import Model, RegressionMLP
import warnings
warnings.filterwarnings("ignore")

input_names = ['input']
output_names = ['output']

pretrained_model = Model.load_from_checkpoint('pretrained_gaze_model.ckpt')

regression_model = RegressionMLP()
regression_model.load_state_dict(torch.load('regression_model.pth'))

person_idx = torch.rand(size=(1, 1)).long()
full_face_image = torch.rand(size=(1, 3, 96, 96)).float()
right_eye_image = left_eye_image = torch.rand(size=(1, 3, 64, 96)).float()
torch.onnx.export(pretrained_model, (person_idx, full_face_image, right_eye_image, left_eye_image),
                              "pretrained_gaze_model.onnx", input_names=input_names,
                              output_names=output_names, verbose='True')

gaze_out = torch.rand(size=(1, 5)).float()
torch.onnx.export(regression_model, gaze_out, "regression_model.onnx", input_names=input_names, output_names=output_names, verbose='True')


