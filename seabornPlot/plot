
from  matplotlib import pyplot as plt
import  numpy as np
import seaborn as sns
import pandas as pd

trainDiabete=pd.read_csv('../files/train.csv', delimiter=',')
result3P = pd.read_csv('../files/optimization3.csv', delimiter=',')
result6P = pd.read_csv('../files/optimization6.csv', delimiter=',')
result15P = pd.read_csv('../files/optimization15.csv', delimiter=',')


def catplot(xAttribute, classLabel, hueAttribute ,dataset):
    sns.set(style="whitegrid")
    # Draw a nested barplot to show survival for class and sex
    g = sns.catplot(x=xAttribute, y=classLabel, hue=hueAttribute, data=dataset,
                height=6, kind="bar", palette="muted")
    g.despine(left=True)
    g.set_ylabels("survival probability")
    plt.show()


# titanic = sns.load_dataset("titanic")
# catplot("class", "survived", "sex", titanic)
#---------------------------------------------------------------------------
def hitmap(dataset, *pivotAttribute):
    sns.set()
    # Load the example flights dataset and conver to long-form
    # flights_long = sns.load_dataset("flights")
    flights = dataset.pivot("month", "year", "passengers")
    # flights = dataset.pivot(pivotAttribute)
    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)
    plt.show()



# flights_long = sns.load_dataset("flights")
# pivotAttribute="month", "year", "passengers"
# hitmap(flights_long )
# print(flights_long.head(3))
#----------------------------------------------------------------------------------
def plotPivot(dataset,xName):
    pivot = dataset.pivot_table(index=xName, values='price', aggfunc=np.median)
    pivot.plot(kind='bar', color='blue')
    plt.xlabel(xName)
    plt.ylabel('Median Price')
    plt.xticks(rotation=0)
    plt.show()

#----------------------------------------Perfect barchart for result------------------------------------------
def resultBarPlot(paletteColor):
    result=pd.read_csv('D:/Research/Experiment/plot1.csv', delimiter=',')
    ax = sns.barplot(x=result['Name'], palette=paletteColor,y=result['LDA'],data=result)
    # sns.color_palette("Blues")
    plt.show()
# --------------------------------------------------------------------------------------------------------------
def listOfPalette():
    palettes=["rocket","BuGn_r","vlag","deep","Blues","GnBu_d","PuBuGn_d"]
    for i in palettes:
        print(i)

# listOfPalette()
#------------------------------- 4 plot in one page----------------------------------------
def resultBarPlot2in1(paletteColor):
    result=pd.read_csv('D:/Research/Experiment/plot1.csv', delimiter=',')
    plt.figure(1)
    # subplot(nrows, ncols, index, **kwargs)
    # plt.subplot(211)
    plt.subplot(221)
    ax1 = sns.barplot(x=result['Name'], palette=paletteColor,y=result['LDA'],data=result)
    ax1.set_xticklabels(rotation=30,labels=result)
    plt.plot()
    plt.title('Plot shomare 1')
    plt.subplot(222)
    ax2 = sns.barplot(x=result['Name'], palette=paletteColor,y=result['LDAT'],data=result)
    ax2.set_xticklabels(rotation=30, labels=result)
    plt.plot()
    plt.title('Plot shomare 2')
    plt.subplot(223)
    ax1 = sns.barplot(x=result['Name'], palette=paletteColor, y=result['LDA'], data=result)
    ax1.set_xticklabels(rotation=30, labels=result)
    plt.plot()
    plt.title('Plot shomare 3')
    plt.subplot(224)
    ax1 = sns.barplot(x=result['Name'], palette=paletteColor, y=result['LDA'], data=result)
    ax1.set_xticklabels(rotation=30, labels=result)
    plt.plot()
    plt.title('Plot shomare 4')
    plt.show()
# ************************** line plot perfect !!! *****************************
def lineplot( xLable, yLable):
    plt.rcParams.update({'font.family' : 'serif','font.weight': 'light', 'font.size': 15})
    result3P = pd.read_csv('../files/optimization3.csv', delimiter=',')
    plt.plot(result3P['Party3'],result3P['Linear'], marker='o',color='red')
    plt.plot(result3P['Party3'],result3P['Linear&DP'], marker='+',color='gold')
    plt.plot(result3P['Party3'],result3P['Tree'], marker='8', color='b')
    plt.plot(result3P['Party3'],result3P['Tree&DP'], marker='x',color='g')
    plt.xlabel(xLable)
    plt.ylabel(yLable)
    plt.grid('on', 'both')
    plt.legend()
    plt.show()

# -------------------------------------------------------------------------------
def lineplot2():
    result3P = pd.read_csv('../files/optimization3.csv', delimiter=',')
    # multiple line plot
    x=result3P['Party3']
    y1=result3P['Linear']
    y2=result3P['Linear&DP']
    y3=result3P['Tree']
    y4=result3P['Tree&DP']
    plt.plot(x, y1, marker='o', markerfacecolor='k', markersize=8, color='grey', linewidth=3, label="Linear")
    plt.plot(x, y2, marker='8',markerfacecolor='k',  markersize=15, color='grey',  linewidth=3, label="Linear & DP")
    plt.plot(x, y3, marker='*',markerfacecolor='k',  markersize=8,  color='grey', linewidth=3, linestyle='--', label="Tree")
    plt.plot(x, y4, marker='x', markerfacecolor='k',  markersize=8, color='grey', linewidth=3, label="DP & Tree")
    plt.grid('on', 'both')
    plt.legend(loc = 'lower right')
    plt.show()

# lineplot2()

# ----------------------------------------------------------------------------
# it draw one plot with many lines
def lineplot3():
    f = plt.figure()
    sns.set_context('poster')
    sns.lineplot(x='Instance (K)', y='Time (Sec)', hue='Method', data=result3P, palette="ch:2.5,.25")
    sns.set(rc={'figure.figsize': (10, 10)})
    plt.grid('on', 'both')
    plt.legend(loc='lower right')
    sns.axes_style("darkgrid")
    plt.show()

# lineplot3()
# ----------------------------------------------------------------------------

def lineplot4():
    plt.figure(1)
    plt.subplot(121)
    sns.set_context(rc={"lines.linewidth":"2.5"})
    sns.lineplot(x='Instance (K)', y='Time (Sec)', hue='Method',style='Method',
                 data=result3P, palette=["Red","gold", "Blue","Green"],markers=True)
    # sns.set_context('poster')
    # sns.set(rc={'figure.figsize': (10, 10)})
    plt.grid('on', 'both')
    plt.legend(loc='upper left')
    plt.title('3 Party')
    plt.plot()
    plt.subplot(122)
    sns.lineplot(x='Instance (K)', y='Time (Sec)', hue='Method', style='Method',
                 data=result15P, palette=["Red","gold", "Blue","Green"],markers=True)
    # sns.set_context('poster')
    # sns.set(rc={'figure.figsize': (10, 10)})
    plt.grid('on', 'both')
    plt.legend(loc='upper left')
    plt.title('15 Party')
    plt.plot()

    plt.show()

# lineplot4()
# ----------------------------------------------------------------------------

# palette1="rocket"
# palette2="BuGn_r"
# palette3="vlag"
# palette4="deep"
# palette5="Blues"
# palette6="GnBu_d"
def lineplot5():
    plt.figure(1)
    plt.subplot(131)
    # plt.subplot(311)
    sns.set_context(rc={"lines.linewidth":"2.5"})
    # sns.lineplot(x='Instance (K)', y='Time (Sec)', hue='Method',data=result3P,style='Method', markers=True, palette="ch:2.5,.25")
    sns.lineplot(x='Instance (M)', y='Time (Sec)', hue='Method',data=result3P,style='Method', markers=True, palette="rocket")
    # sns.set_context('poster')
    # sns.set(rc={'figure.figsize': (10, 10)})
    plt.grid('on', 'both')
    plt.legend(loc='upper left')
    plt.title('3P')
    plt.plot()
    plt.subplot(132)
    # plt.subplot(312)
    # sns.lineplot(x='Instance (K)', y='Time (Sec)', hue='Method', data=result6P, style='Method', markers=True,palette="ch:2.5,.25")
    sns.lineplot(x='Instance (M)', y='Time (Sec)', hue='Method', data=result6P, style='Method', markers=True,palette="rocket")
    # sns.set_context('poster')
    # sns.set(rc={'figure.figsize': (10, 10)})
    plt.grid('on', 'both')
    plt.legend(loc='upper left')
    plt.title('6P')
    plt.plot()
    plt.subplot(133)
    # plt.subplot(313)
    # sns.lineplot(x='Instance (K)', y='Time (Sec)', hue='Method',data=result15P,style='Method',markers=True, palette="ch:2.5,.25")
    sns.lineplot(x='Instance (M)', y='Time (Sec)', hue='Method',data=result15P,style='Method',markers=True, palette="rocket")
    # sns.set_context('poster')
    # sns.set(rc={'figure.figsize': (10, 10)})
    plt.grid('on', 'both')
    plt.legend(loc='upper left')
    plt.title('15P')
    plt.plot()

    plt.show()


# lineplot5()

# ------------------------------------------------------------------------------------------------

def lineplot6():
    df = pd.read_csv("../files/LDA.csv", delimiter=',')
    x = df['Records']
    y = df['Time']
    plt.grid('on', 'both')
    plt.xlabel("Records (Million)")
    plt.ylabel("Times (Second)")
    plt.plot(x, y, marker='o', markerfacecolor='k', markersize=8, color='blue', linewidth=3)
    plt.show()
# --------------------------------------------------------------------------------------------------

def lineplot7():
    df = pd.read_csv("../files/LDA.csv", delimiter=',')
    x1 = df['Instance (Million)']
    y1 = df['Time (Sec)']
    plt.grid('on', 'both')
    sns.lineplot(data=df,x=x1,y=y1,color="firebrick", linewidth=3)
    plt.show()

# lineplot7()
# ------------------------------------------------------------------------------------------------------

def scatterplotOfDataset():
    sns.pairplot(trainDiabete,
                 vars=['pregnancyHistory', 'plasma', 'bloodPressure', 'skinThickness', 'insulin', 'bodyMass',
                       'pedigree', 'age'], hue='class', markers=["o", "s"], palette="GnBu_d")
    # plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show()


# scatterplotOfDataset()
# -------------------------------------------------------------------------------------------------------------

def biaplotOfDataset():
    path = '../files/train.csv'
    df = pd.read_csv(path, delimiter=',')
    ax = sns.scatterplot(x=df['insulin'], y=df['plasma'], hue=df['class'], style=df['class'])
    sns.scatterplot(x=df['insulin'], y=df['bloodPressure'], hue=df['class'], style=df['class'])
    sns.scatterplot(x=df['insulin'], y=df['skinThickness'], hue=df['class'], style=df['class'])
    sns.scatterplot(x=df['insulin'], y=df['pregnancyHistory'], hue=df['class'], style=df['class'])
    sns.scatterplot(x=df['insulin'], y=df['bodyMass'], hue=df['class'], style=df['class'])
    sns.scatterplot(x=df['insulin'], y=df['pedigree'], hue=df['class'], style=df['class'])
    sns.scatterplot(x=df['insulin'], y=df['age'], hue=df['class'], style=df['class'])
    ax.get_legend().set_visible(False)
    plt.xlabel('')
    plt.ylabel('')
    plt.show()

# scatterplotOfDataset()
# --------------------------------------------------------------------------------------------------------------------
# pie chart
# import the pyplot library

import matplotlib.pyplot as plotter

# pieLabels = 'Female/Single', 'Female/Married', 'Male/Single', 'Male/Married',
# populationShare = [70,  3, 678, 18]
# figureObject, axesObject = plotter.subplots()
# cmap = plt.get_cmap('Spectral')
# colors = [cmap(i) for i in np.linspace(0, 1, 8)]
# axesObject.pie(populationShare, labels=pieLabels,autopct='%1.2f',startangle=90,colors=colors)
# axesObject.axis('equal')

# plotter.show()
# --------------------------------------------------------------------------
# pie chart 2


counts = pd.Series([70,  3, 678, 18], index=['Female/Single', 'Female/Married', 'Male/Married', 'Male/Single'])
# counts = pd.Series([386, 114, 390, 483, 702, 791], index=['0_4', '4_8', '8_12', '12_16', '16_20', '20_24'])

# explode = (0, 0, 0, 0,0, 0)
explode = (0.1, 0.1, 0.1, 0.1)
# colors = ['#191970', '#001CF0', '#0038E2', '#0055D4', '#008DB8', '#00AAAA']
colors = ['#191970','#00FF80', '#001CF0', '#00AAAA'   ]
# colors = ['#191970', '#001CF0', '#0038E2', '#0055D4', '#0071C6', '#008DB8', '#00AAAA', '#00C69C', '#00E28E', '#00FF80' ]
counts.plot(kind='pie', fontsize=17,autopct='%1.2f', colors=colors, explode=explode)
plt.axis('equal')
# plt.ylabel('Programming Time')
plt.ylabel('')
plt.legend(labels=counts.index, loc="best")
plt.show()
# --------------------------------------------------------------------------

