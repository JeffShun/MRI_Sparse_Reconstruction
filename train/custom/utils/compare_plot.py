import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd


def data_parser(csv_path, metric):
    df = pd.read_csv(csv_path)
    return list(df[metric])


def plot_box(datas, labels):

    # 创建柱状图
    fig = go.Figure()

    # 创建子图
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08)

    colors = ['rgb(0,120,40)', 'rgb(40,40,140)', 'rgb(50,100,200)','rgb(200,100,90)','rgb(100,220,90)','rgb(200,30,90)']

    fontsize_small = 50
    fontsize_mid = 65
    fontsize_big = 80

    for i in range(len(labels)):
        # 添加PSNR数据
        fig.add_trace(go.Box(y=datas["psnr"][i], name=labels[i], boxpoints='all', jitter=0.3, pointpos=-1.8, marker=dict(color=colors[i]), showlegend=False), row=1, col=1)

        fig.add_trace(go.Box(y=datas["ssim"][i], name=labels[i], boxpoints='all', jitter=0.3, pointpos=-1.8, marker=dict(color=colors[i]), showlegend=False), row=1, col=2)

        fig.add_trace(go.Box(y=np.array(datas["nmse"][i])*1000, name=labels[i], boxpoints='all', jitter=0.3, pointpos=-1.8, marker=dict(color=colors[i]), showlegend=False), row=1, col=3)


    layout = go.Layout(
        title=' 不同模型重建性能比较',
        title_x=0.50,
        margin=dict(
            t=300, 
            b=300,  
            l=200
        ),
        legend=dict(
            font=dict(size=fontsize_big, family="Times New Roman"),
            x=1.05,  # 控制legend的x位置
            y=0.97,  # 控制legend的y位置
            bgcolor='rgba(0, 0, 0, 0)',  # 设置图例的背景颜色为透明,
        ),
        width=5500,  # 设置图表的宽度
        height=2800,  # 设置图表的高度
        plot_bgcolor='rgba(0, 0, 0, 0)',
        titlefont=dict(size=fontsize_big, family="Times New Roman"),
    )
    fig.update_layout(layout)

    # 设置PSNR子图的布局
    fig.update_xaxes(showline=True, linecolor='black', tickangle=30, linewidth=2, mirror=True, showgrid=False, gridwidth=1, gridcolor='grey',title='',  title_text="",tickfont=dict(size=fontsize_mid,family="Times New Roman"), titlefont=dict(size=fontsize_big, family="Times New Roman"), row=1, col=1, griddash='dash')
    fig.update_yaxes(showline=True, linecolor='black', linewidth=2, mirror=True, showgrid=True, dtick=0.1, gridwidth=1, gridcolor='grey',title='PSNR', tickfont=dict(size=fontsize_small,family="Times New Roman"), titlefont=dict(size=fontsize_big, family="Times New Roman"), range=[25, 45], row=1, col=1, tickmode='array', tickvals=list(np.arange(25, 46, 1)), griddash='dash')

    # 设置SSIM子图的布局
    fig.update_xaxes(showline=True, linecolor='black', tickangle=30,linewidth=2, mirror=True, showgrid=False, gridwidth=1, gridcolor='grey',title='', title_text="",tickfont=dict(size=fontsize_mid,family="Times New Roman"), titlefont=dict(size=fontsize_big, family="Times New Roman"), row=1, col=2, griddash='dash')
    fig.update_yaxes(showline=True, linecolor='black', linewidth=2, mirror=True, showgrid=True, dtick=0.01, gridwidth=1, gridcolor='grey',title='SSIM', tickformat=".2f", tickfont=dict(size=fontsize_small,family="Times New Roman"),titlefont=dict(size=fontsize_big, family="Times New Roman"), range=[0.65, 1.0], row=1, col=2, tickmode='array', tickvals=list(np.arange(0.64, 1.01, 0.01)), griddash='dash')

    # 设置NMSE子图的布局
    fig.update_xaxes(showline=True, linecolor='black', tickangle=30,linewidth=2, mirror=True, showgrid=False, gridwidth=1, gridcolor='grey',title='', title_text="", tickfont=dict(size=fontsize_mid,family="Times New Roman"), titlefont=dict(size=fontsize_big, family="Times New Roman"), row=1, col=3, griddash='dash')
    fig.update_yaxes(showline=True, linecolor='black', linewidth=2, mirror=True, showgrid=True, dtick=0.1, gridwidth=1, gridcolor='grey',title='NMSE(1e-3)', tickfont=dict(size=fontsize_small,family="Times New Roman"),titlefont=dict(size=fontsize_big, family="Times New Roman"), range=[0, 35], row=1, col=3, tickmode='array', tickvals=list(np.arange(0, 37, 1)), griddash='dash')

    # 显示图表
    fig.show()

if __name__ == "__main__":
    compare_models = ["Paper-MW-ResUnet-DC", "Paper-MW-ResUnet-DC-LD","Paper-MW-ResUnet-DC4","Paper-MW-ResUnet-DC4-LD","Paper-MW-ResUnet-DC8","Paper-MW-ResUnet-DC8-LD"]
    datas_psnr = []
    datas_ssim = []
    datas_nmse = []
    for model in compare_models:
        csv_path = "../example/data/output/%s/metric_data/metrics.csv"%(model)
        datas_psnr.append(data_parser(csv_path, "PSNR"))
        datas_ssim.append(data_parser(csv_path, "SSIM"))
        datas_nmse.append(data_parser(csv_path, "NMSE"))
    plot_box({"psnr":datas_psnr, "ssim":datas_ssim,"nmse":datas_nmse}, compare_models)