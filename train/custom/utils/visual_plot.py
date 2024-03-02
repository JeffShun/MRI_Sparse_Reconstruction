import plotly.graph_objs as go
import pandas as pd


def data_parser(csv_path, metric):
    df = pd.read_csv(csv_path)
    return list(df[metric])

def plot_box(datas, labels):
    traces = []
    for i in range(len(datas)):
        traces.append(go.Box(y=datas[i], name=labels[i], boxpoints='all', jitter=0.3, pointpos=-1.8))
    # 创建布局
    layout = go.Layout(
        title='不同模型SSIM指标比较（Mini Dataset）',
        title_x=0.5,
        xaxis=dict(title='', tickfont=dict(size=20),titlefont=dict(size=20)),
        yaxis=dict(title='SSIM', tickfont=dict(size=20),titlefont=dict(size=20)),
        legend=dict(
            font=dict(size=12)
        ),
        titlefont=dict(size=20)
    )
    # 创建图表对象
    fig = go.Figure(traces, layout=layout)
    fig.show()

if __name__ == "__main__":
    compare_models = ["DUNet-DCFree_mini", "DUNet-GD_mini","DUNet-PM-LS_mini","DUNet-PM-LSW_mini","ResUnet-PM-LSW_mini","ResUnet-PM-LSW-AUG_mini"]
    datas = []
    for model in compare_models:
        csv_path = "../example/data/output/%s/metric_data/metrics.csv"%(model)
        ssim = data_parser(csv_path, "SSIM")
        datas.append(ssim)
    plot_box(datas,compare_models)
