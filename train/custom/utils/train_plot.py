import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# 模型名称
x_axis_PSNR = ['Iterations']
x_axis_NMSE = ['Iterations']
x_axis_SSIM = ['Iterations']

# 训练结果和测试结果，每组只有三个数据

model_1_PSNR = [33.88] 
model_2_PSNR = [34.51]
model_3_PSNR = [34.79]

model_1_SSIM = [0.8713] 
model_2_SSIM = [0.8736]
model_3_SSIM = [0.8781]

model_1_NMSE = [13.3] 
model_2_NMSE = [12.7]
model_3_NMSE = [12.1]

# 创建柱状图
fig = go.Figure()

# 创建子图
fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.11)

color1, color2, color3 = 'rgb(0, 120, 40)', 'rgb(50, 100, 200)','rgb(200, 100, 90)'

fontsize_small = 140
fontsize_big = 180

# 添加第一组训练和测试结果的柱状图
fig.add_trace(go.Bar(x=x_axis_PSNR, y=model_1_PSNR, name='C1', text=model_1_PSNR, textposition='outside', marker=dict(color=color1), textfont=dict(size=fontsize_small,family="Times New Roman"), showlegend=False), row=1, col=1)
fig.add_trace(go.Bar(x=x_axis_PSNR, y=model_2_PSNR, name='C4', text=model_2_PSNR, textposition='outside', marker=dict(color=color2), textfont=dict(size=fontsize_small,family="Times New Roman"), showlegend=False), row=1, col=1)
fig.add_trace(go.Bar(x=x_axis_PSNR, y=model_3_PSNR, name='C8', text=model_3_PSNR, textposition='outside', marker=dict(color=color3), textfont=dict(size=fontsize_small,family="Times New Roman"), showlegend=False), row=1, col=1)

fig.add_trace(go.Bar(x=x_axis_SSIM, y=model_1_SSIM, name='C1', text=model_1_SSIM, textposition='outside', marker=dict(color=color1), textfont=dict(size=fontsize_small,family="Times New Roman"), showlegend=False), row=1, col=2)
fig.add_trace(go.Bar(x=x_axis_SSIM, y=model_2_SSIM, name='C4', text=model_2_SSIM, textposition='outside', marker=dict(color=color2), textfont=dict(size=fontsize_small,family="Times New Roman"), showlegend=False), row=1, col=2)
fig.add_trace(go.Bar(x=x_axis_SSIM, y=model_3_SSIM, name='C8', text=model_3_SSIM, textposition='outside', marker=dict(color=color3), textfont=dict(size=fontsize_small,family="Times New Roman"), showlegend=False), row=1, col=2)

fig.add_trace(go.Bar(x=x_axis_NMSE, y=model_1_NMSE, name='C1', text=model_1_NMSE, textposition='outside', marker=dict(color=color1), textfont=dict(size=fontsize_small,family="Times New Roman"), showlegend=False), row=1, col=3)
fig.add_trace(go.Bar(x=x_axis_NMSE, y=model_2_NMSE, name='C4', text=model_2_NMSE, textposition='outside', marker=dict(color=color2), textfont=dict(size=fontsize_small,family="Times New Roman"), showlegend=False), row=1, col=3)
fig.add_trace(go.Bar(x=x_axis_NMSE, y=model_3_NMSE, name='C8', text=model_3_NMSE, textposition='outside', marker=dict(color=color3), textfont=dict(size=fontsize_small,family="Times New Roman"), showlegend=False), row=1, col=3)

# 使用Scatter来模拟大方块
fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers+lines', line=dict(width=10, color=color1), marker=dict(size=16, symbol='circle', opacity=0), name='C1', legendgroup="C1"), row=1, col=3)
fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers+lines', line=dict(width=10, color=color2), marker=dict(size=16, symbol='circle', opacity=0), name='C4', legendgroup="C2"), row=1, col=3)
fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers+lines', line=dict(width=10, color=color3), marker=dict(size=16, symbol='circle', opacity=0), name='C8', legendgroup="C3"), row=1, col=3)


layout = go.Layout(
    title=' Effect of iteration number',
    title_x=0.52,
    margin=dict(
        t=300, 
        b=300,  
        l=500
    ),
    legend=dict(
        font=dict(size=fontsize_small),
        x=1.03,  # 控制legend的x位置
        y=0.95,  # 控制legend的y位置
        bgcolor='rgba(0, 0, 0, 0)',  # 设置图例的背景颜色为透明,
    ),
    width=6500,  # 设置图表的宽度
    height=3800,  # 设置图表的高度
    plot_bgcolor='rgba(0, 0, 0, 0)',
    titlefont=dict(size=fontsize_big, family="Times New Roman"),
)
fig.update_layout(layout)

# 设置PSNR子图的布局
fig.update_xaxes(showline=True, linecolor='black', linewidth=2, mirror=True, showgrid=False, gridwidth=1, gridcolor='darkgrey',title='',  title_text="",tickfont=dict(size=fontsize_small,family="Times New Roman"), titlefont=dict(size=fontsize_small, family="Times New Roman"), row=1, col=1)
fig.update_yaxes(showline=True, linecolor='black', linewidth=2, mirror=True, showgrid=True, dtick=0.1, gridwidth=2, gridcolor='darkgrey',title='PSNR', tickfont=dict(size=fontsize_small,family="Times New Roman"),tickformat=".1f", titlefont=dict(size=fontsize_small, family="Times New Roman"), range=[33, 35], row=1, col=1, tickmode='array', tickvals=list(np.arange(33, 35.2, 0.2)))

# 设置SSIM子图的布局
fig.update_xaxes(showline=True, linecolor='black', linewidth=2, mirror=True, showgrid=False, gridwidth=1, gridcolor='darkgrey',title='', title_text="",tickfont=dict(size=fontsize_small,family="Times New Roman"), titlefont=dict(size=fontsize_small, family="Times New Roman"), row=1, col=2)
fig.update_yaxes(showline=True, linecolor='black', linewidth=2, mirror=True, showgrid=True, dtick=0.001, gridwidth=2, gridcolor='darkgrey',title='SSIM', tickfont=dict(size=fontsize_small,family="Times New Roman"), tickformat=".3f",titlefont=dict(size=fontsize_small, family="Times New Roman"), range=[0.865, 0.881], row=1, col=2, tickmode='array', tickvals=list(np.arange(0.865, 0.883, 0.002)))

# 设置NMSE子图的布局
fig.update_xaxes(showline=True, linecolor='black', linewidth=2, mirror=True, showgrid=False, gridwidth=1, gridcolor='darkgrey',title='', title_text="", tickfont=dict(size=fontsize_small,family="Times New Roman"), titlefont=dict(size=fontsize_small, family="Times New Roman"), row=1, col=3)
fig.update_yaxes(showline=True, linecolor='black', linewidth=2, mirror=True, showgrid=True, dtick=0.1, gridwidth=2, gridcolor='darkgrey',title='NMSE(1e-3)', tickfont=dict(size=fontsize_small,family="Times New Roman"), tickformat=".1f",titlefont=dict(size=fontsize_small, family="Times New Roman"), range=[11, 13.8], row=1, col=3, tickmode='array', tickvals=list(np.arange(11, 13.8, 0.4)))

# 显示图表
fig.show()

