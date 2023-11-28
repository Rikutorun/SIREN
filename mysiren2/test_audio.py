import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from torch.utils.data import DataLoader
import dataio_test, meta_modules, utils, training, loss_functions, modules
import torch
import configargparse
import numpy as np
from matplotlib import pyplot as plt

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=False, default = './test', 
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')


# General training options
p.add_argument('--model', type=str, default='sine', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'],
               help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--velocity', type=str, default='uniform', required=False, choices=['uniform', 'square', 'circle'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')

p.add_argument('--checkpoint_path', default='./logs/experiment/checkpoints/model_final.pth', help='Checkpoint to trained model.')
opt = p.parse_args()


# if we have a velocity perturbation, offset the source
if opt.velocity!='uniform':
    source_coords = [-0.35, 0.]
else:
    source_coords = [0., 0.]

dataset = dataio_test.SingleHelmholtzSource(sidelength = 128, velocity = opt.velocity, source_coords = source_coords)

dataloader = DataLoader(dataset, shuffle=True, batch_size=32, pin_memory=True, num_workers=0)

model = modules.SingleBVPNet(out_features=2, type=opt.model, mode=opt.mode, final_layer_factor=1.)

model.load_state_dict(torch.load(opt.checkpoint_path))
model.cuda()

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

# Get ground truth and input data
model_input, gt = next(iter(dataloader))
model_input = {key: value.cuda() for key, value in model_input.items()}
gt = {key: value.cuda() for key, value in gt.items()}

# Evaluate the trained model
with torch.no_grad():
    model_output = model(model_input)
    
model_out = model_output['model_out'].to('cpu').detach().numpy().copy()
model_fig = np.real(model_out).reshape(-1,2)[:,0]
model_line_integral = np.sum(model_fig.reshape(-1,128),axis = 1)
model_line_integral = np.expand_dims(model_line_integral, axis = 0)
fig = plt.imshow(np.real(model_fig.reshape(-1,128)))
plt.show()
fig_line_integral = plt.imshow(model_line_integral)
plt.show()

#waveform = torch.squeeze(model_output['model_out']).detach().cpu().numpy()
#rate = torch.squeeze(gt['rate']).detach().cpu().numpy()
#wavfile.write(os.path.join(opt.logging_root, opt.experiment_name, 'pred_waveform.wav'), rate, waveform)