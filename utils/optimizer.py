import torch

class Optimizer(object):
    def __init__(self, model_params, optimizer_name="Adam", lr=1e-3, weight_decay=0):
        if optimizer_name=="Adam":
            self.optimizer = torch.optim.Adam(model_params, 
                                         lr=lr, 
                                         betas=(0.9, 0.999), 
                                         eps=1e-08, 
                                         weight_decay=weight_decay,
                                         amsgrad=True)
        elif optimizer_name=="SGD":
            self.optimizer = torch.optim.SGD(model_params, 
                                        lr=lr, 
                                        momentum=0, 
                                        dampening=0, 
                                        weight_decay=weight_decay,
                                        nesterov=False)
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def step(self):
        self.optimizer.step()
    
    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
        
    def state_dict(self):
        return self.optimizer.state_dict()