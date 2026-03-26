import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

kendall_file = "kendall.xlsx"
kendall_sheet = pd.read_excel(kendall_file, sheet_name="kendall")
metric_names = ["Organ DSC", "Organ NSD", "Lesion DSC", "Lesion NSD", "GPU", "Time"]
colors_dict = {'Organ DSC': 'rgb(239, 85, 59)',
               'Organ NSD': 'rgb(171, 99, 250)',
               'Lesion DSC': 'rgb(255, 0, 255)',
               'Lesion NSD': 'rgb(0, 0, 255)',
               'GPU Memory': 'rgb(255, 128, 0)',
               'Runtime': 'rgb(0, 255, 0)'}

series = []
values = []

OrganDSCMean_Rank = []
for index, row in kendall_sheet.iterrows():
    OrganDSCMean_Rank.append(row["Organ DSC"])
values.extend(OrganDSCMean_Rank)
series.extend(['Organ DSC', ] * len(OrganDSCMean_Rank))
    
OrganNSDMean_Rank = []
for index, row in kendall_sheet.iterrows():
    OrganNSDMean_Rank.append(row["Organ NSD"])
values.extend(OrganNSDMean_Rank)
series.extend(['Organ NSD', ] * len(OrganNSDMean_Rank))

Lesion_DSC_Rank = []
for index, row in kendall_sheet.iterrows():
    Lesion_DSC_Rank.append(row["Lesion DSC"])
values.extend(Lesion_DSC_Rank)
series.extend(['Lesion DSC', ] * len(Lesion_DSC_Rank))

Lesion_NSD_Rank = []
for index, row in kendall_sheet.iterrows():
    Lesion_NSD_Rank.append(row["Lesion NSD"])
values.extend(Lesion_NSD_Rank)
series.extend(['Lesion NSD', ] * len(Lesion_NSD_Rank))

RankAUC_GPU_Time = []
for index, row in kendall_sheet.iterrows():
    RankAUC_GPU_Time.append(row["GPU Memory"])
values.extend(RankAUC_GPU_Time)
series.extend(['GPU Memory', ] * len(RankAUC_GPU_Time))

RealTime_Rank = []
for index, row in kendall_sheet.iterrows():
    RealTime_Rank.append(row["Runtime"])
values.extend(RealTime_Rank)
series.extend(['Runtime', ] * len(RealTime_Rank))
    
dic = {"series": series, "values": values}
df = pd.DataFrame(dic)

fig = go.Figure()
for category in df['series'].unique():
    fig.add_trace(go.Violin(x=df['series'][df['series'] == category],
                            y=df['values'][df['series'] == category]))

fig.add_trace(go.Box(x=df['series'],
                     y=df['values'],
                     width=0.1,
                     fillcolor="lightgray",
                     line=dict(width=0.8, color='black')))

fig.update_layout(showlegend=False)
fig.update_yaxes(range=[-0.1, 1.1])
fig.update_layout(
    font=dict(family='times new roman', size=17, color="#000000"),
    width=800,
    height=550,
    xaxis=dict(title=""),
    yaxis=dict(title="Kendall's tau",
               tickvals=[0, 0.25, 0.5, 0.75, 1],
               ticktext=["0.00", "0.25", "0.50", "0.75", "1.00"],
               showticklabels=True),
    title=dict(text='')
)

fig.update_layout(margin=dict(l=0, r=0, b=0, t=25))
# fig.show()
fig.write_image("fig2c_kendall.png", scale=2)
