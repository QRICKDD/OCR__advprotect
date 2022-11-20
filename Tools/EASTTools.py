import torch
from model_EAST.model import EAST



def load_EASTmodel(model_name = 'F:\OCR-TASK\OCR__advprotect\model_EAST\weight\east_vgg16.pth'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST(False).to(device)
    model.load_state_dict(torch.load(model_name))
    model.eval()
    return model
