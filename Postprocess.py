import forgi.graph.bulge_graph as fgb
import forgi.visual.mplotlib as fvm
import collections
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def tnp(t):
    return t.cpu().detach().numpy()


def test_one(M, dataset, idx, return_structure=False):
    x, y1, y2, y3, L, idx = dataset[idx]
    device = M.device
    structure = dataset.get_structures(idx)[0]
    o1, o2, o3, emb = M.forward(x.to(device))
    loss = M.Losses[-1](o3, y3)
    if return_structure:
        return loss, y3, o3, y1, o1, y2, o2, structure
    return loss, y3, o3

def plot_result(Y_hat, Y, targets, first_k=68, err=None, structure=None, target_num=0):
    fig = make_subplots(rows=2, cols=1)
    marker_dict = dict(color=encode_color(structure[:first_k])) if structure else {}
    fig.add_trace(go.Bar(y=tnp(Y[target_num][:first_k]),
                           error_y=dict(type='data', symmetric=True, array=err),
                           name='Ground Truth', marker=marker_dict), row=1, col=1)
    fig.add_trace(go.Bar(y=tnp(Y_hat[target_num][:first_k]),
                           name='Predicted', marker=marker_dict), row=2, col=1)
    fig.update_layout(height=600, width=800, title_text=f"{targets[target_num]} Ground Truth vs Predicted <br>\
    colored by loop type")
    fig.show()