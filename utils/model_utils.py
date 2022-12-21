def toggle_grad(model, mode):
    for param in model.parameters():
        param.requires_grad = mode
    return model

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

                  
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)