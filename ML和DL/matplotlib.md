# Matplotlib画图

- [](#)
	- [用bar绘制垂直条形图](#用bar绘制垂直条形图)
	- [用hist绘制柱状图](#用hist绘制柱状图)
	- [plot曲线图](#plot曲线图)
	- [scatter绘制散点图](#scatter绘制散点图)
	- [bar绘制垂直条形图和barh绘制水平条形图](#bar绘制垂直条形图和barh绘制水平条形图)
	- [fill和fill_between绘制填充图和stackplot绘堆叠图](#fill和fill_between绘制填充图和stackplot绘堆叠图)
	- [hist绘制柱状图和boxplot绘制箱线图](#hist绘制柱状图和boxplot绘制箱线图)
	- [imshow显示二维图形](#imshow显示二维图形)
	- [contour+contourf绘制等高线](#contour+contourf绘制等高线)
	- [Axes3D绘制三维曲线图](#Axes3D绘制三维曲线图)
	- [subplot绘制多图](#subplot绘制多图)	
	- [figure绘制多图](#figure绘制多图)	
	- [figure图的嵌套](#figure图的嵌套)	
	- [主次坐标轴](#主次坐标轴)	
	
	
	
## 用bar绘制垂直条形图
+ **垂直条形图**
```python

x=[5,8,10]
y=[12,16,6]
std = [5,6,7]
x2=[6,9,11]
y2=[6,15,7]
std2 = [1,2,3]
plt.bar(x,y,yerr=std )
plt.bar(x2,y2,yerr=std2)
plt.ylabel("Y axis")
plt.xlabel("X axis")

for i, j in zip(x, y):
    plt.text(i, j+1, '%d'%j)

for i, j in zip(x2, y2):
    plt.text(i, j+1, '%d'%j)

plt.show()

```
![垂直条形图](https://i.loli.net/2019/04/14/5cb2cf17532e8.png)

```python
n = 12
X = np.arange(n)
Y1 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)
Y2 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)

plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

for x,y in zip(X,Y1):
    plt.text(x, y+0.05, '%.2f' % y) # 对应位置显示大小

plt.ylim(-1.25,+1.25)
plt.show()

```
![垂直条形图](https://i.loli.net/2019/04/14/5cb2cf68a4e54.jpg)


+ **堆叠条形图**
```python
x=[5,8,10]
y=[12,16,6]
std = [5,6,7]

y2=[6,15,7]
std2 = [1,2,3]
plt.bar(x,y,yerr=std)
plt.bar(x,y2,bottom=y,yerr=std2) # 以y中的值为底
plt.ylabel("Y axis")
plt.xlabel("X axis")
plt.legend(('y','y2'))
plt.show()

```
![堆叠条形图](https://i.loli.net/2019/04/14/5cb2d7de797e1.jpg)



## 用hist绘制柱状图
```python
a=np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
np.histogram(a,bins = [0,20,40,60,80,100])
hist,bins = np.histogram(a,bins=[0,20,40,60,80,100])
print(hist)
print(bins)
plt.hist(a,bins=[0,20,40,60,80,100])  # bins表示分段，此例中分成5段
plt.title("histogram")
plt.show()

```
![柱状图](https://i.loli.net/2019/04/14/5cb2d2f15a5b3.jpg)


## plot曲线图
```python
x = np.linspace(0,10,100)
fig=plt.figure(figsize=(10,4))
ax0 = fig.add_subplot(1,3,1)
ax1 = fig.add_subplot(1,3,2)
ax2 = fig.add_subplot(1,3,3)

for i in range(1,6):
    ax0.plot(x,i*x)
for i,ls in enumerate(['-','--',':','-.']):
    ax1.plot(x,np.cos(x)+i, linestyle=ls)
for i, (ls,mk) in enumerate(zip(['','-',':'],['o','^','s'])):
    ax2.plot(x,np.cos(x)+i*x, linestyle=ls,marker=mk,markevery=10)

plt.show()

```
![plot曲线图](https://i.loli.net/2019/04/14/5cb2d337a9f26.jpg)


## scatter绘制散点图
```python
x,y,z = np.random.normal(0,1,(3,100))
t = np.arctan2(y,x)
size = 50*np.cos(2*t)**2+10

fig, axes = plt.subplots(1,3,figsize = (10,4))

axes[0].scatter(x,y,marker='o',facecolor='black',s=80)
axes[1].scatter(x,y,s=size,marker='s',color='darkblue') # marker的大小随参数s的大小变化
axes[2].scatter(x,y,c=z,s=size,cmap='gist_ncar') # marker的大小和颜色随参数s和参数c的大小变化
 
plt.show()

```
![scatter绘制散点图](https://i.loli.net/2019/04/14/5cb2d3a6eab85.jpg)




## bar绘制垂直条形图和barh绘制水平条形图
```python
y=[1,3,4,5.5,3,2]
err = [0.2,1,2.5,1,1,0.5]
x = np.arange(len(y))
fig,axes = plt.subplots(1,3,figsize=(10,4))
axes[0].bar(x,y,yerr=err,color='lightblue',ecolor='black')
axes[0].set_ylim([0,10])
axes[0].set_title("bar")

y = np.arange(8)
x1 = y+np.random.random(8)+1
x2 = y+3*np.random.random(8)+1
axes[1].barh(y,x1,color='lightblue')
axes[1].barh(y,-x2,color='salmon')
axes[1].set_title("barh")

left = np.random.randint(0,10,10)
bottom = np.random.randint(0,10,10)
width = np.random.random(10)+0.5
height = np.random.random(10)+0.5
axes[2].bar(left,height, width,bottom,color='salmon')

plt.show()

```
![bar绘制垂直条形图和barh绘制水平条形图](https://i.loli.net/2019/04/14/5cb2d43cf32bf.jpg)


## fill和fill_between绘制填充图和stackplot绘堆叠图
```python
def stackplot_data():
    x = np.linspace(0, 10, 100)
    y = np.random.normal(0, 1, (5, 100))
    y = y.cumsum(axis=1)
    y -= y.min(axis=0, keepdims=True)
    return x, y

def sin_data():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    y2 = np.cos(x)
    return x, y, y2

def fill_data():
    t = np.linspace(0, 2*np.pi, 100)
    r = np.random.normal(0, 1, 100).cumsum()
    r -= r.min()
    return r * np.cos(t), r * np.sin(t)

fig, axes = plt.subplots(1,3,figsize=(10,4))
x, y = fill_data()
axes[0].fill(x,y,color='lightblue')
axes[0].set_title("fill")

x, y1, y2 = sin_data()
err = np.random.rand(x.size)**2+0.1
y = 0.7*x+2
axes[1].fill_between(x,y1,y2,where=y1>y2,color='lightblue')  # y1>y2的区域用颜色lightblue填充
axes[1].fill_between(x,y1,y2,where=y1<y2,color='forestgreen')# y1<y2的区域用颜色forestgreen填充

x, y = stackplot_data()
axes[2].stackplot(x, y.cumsum(axis=0), alpha=0.5)

```
![fill和fill_between绘制填充图、stackplot绘堆叠图](https://i.loli.net/2019/04/14/5cb2d50f4aef0.jpg)


## hist绘制柱状图和boxplot绘制箱线图
```python
def generate_data():
    means = [0, -1, 2.5, 4.3, -3.6]
    sigmas = [1.2, 5, 3, 1.5, 2]
    # Each distribution has a different number of samples.
    nums = [150, 1000, 100, 200, 500]

    dists = [np.random.normal(*args) for args in zip(means, sigmas, nums)]
    return dists

fig, axes = plt.subplots(1,3,figsize=(10,4))
colors = ['cyan','red','blue','green','purple']
dists = generate_data()
np.array(dists).shape
axes[0].set_color_cycle(colors)
for dist in dists:
    axes[0].hist(dist,bins=20,density=True,edgecolor='none',alpha=0.5)
    
result = axes[1].boxplot(dists, patch_artist=True,notch=True,vert=False)
for box, color in zip(result['boxes'],colors):
    box.set(facecolor=color, alpha=0.5)
for item in ['whiskers','caps','medians']:
    plt.setp(result[item],color='gray',linewidth=1.5)
plt.setp(result['fliers'], markeredgecolor='gray', markeredgewidth=1.5)
plt.setp(result['medians'], color='black')
result = axes[2].violinplot(dists, vert=False, showmedians=True)
for body, color in zip(result['bodies'], colors):
    body.set(facecolor=color, alpha=0.5)
for item in ['cbars', 'cmaxes', 'cmins', 'cmedians']:
    plt.setp(result[item], edgecolor='gray', linewidth=1.5)
plt.setp(result['cmedians'], edgecolor='black')
plt.show()

```
![hist绘制柱状图、boxplot绘制箱线图](https://i.loli.net/2019/04/14/5cb2d8be56287.jpg)


## imshow显示二维图形
+ **imshow显示二维图形**
```python
vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
              "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
           "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                    [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                    [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                    [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
                    [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                    [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


fig, ax = plt.subplots(figsize=(10,8))
im = ax.imshow(harvest)

ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))

ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")
plt.tight_layout() ## 自动调整图形大小
plt.show()

```
![imshow显示二维图形](https://i.loli.net/2019/04/14/5cb2d9526a69a.jpg)

+ **imshow显示图形+colorbar**
```python
data = np.random.random((256,256))
fig,ax = plt.subplots()
im = ax.imshow(data,cmap='seismic')
fig.colorbar(im)
plt.show()

```
![imshow显示图形+colorbar](https://i.loli.net/2019/04/14/5cb2da861291a.jpg)

## contour+contourf绘制等高线
```python
def f(x,y):
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

n = 256
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
X,Y = np.meshgrid(x,y)

# 第4个参数用来确定轮廓线\区域的数量和位置，数字4表示使用4个数据间隔；绘制5条轮廓线
plt.contourf(X,Y,f(X,Y),4,alpha=0.75,cmap=plt.cm.hot)  # 用于填充轮廓，不会绘制轮廓线
C = plt.contour(X,Y,f(X, Y),4,color='black',linewidth=0.5)  # 用于绘制轮廓线
plt.clabel(C,inline=True,fontsize=20)  # 给contour的返回对象增加标签到轮廓线上
# 去除横纵坐标
plt.xticks(())
plt.yticks(())
plt.show()

```
![contour+contourf绘制等高线](https://i.loli.net/2019/04/14/5cb2daf733f29.jpg)



## Axes3D绘制三维曲线图
```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

n=256
x=np.arange(-4,4,0.25)
y=np.arange(-4,4,0.25)

X,Y = np.meshgrid(x,y)
R = np.sqrt(X**2+Y**2)
Z = np.sin(R)

ax.plot_surface(X,Y,Z,cmap=plt.get_cmap('rainbow'))
ax.contour(X,Y,Z,offset=-2,cmap='rainbow') # 绘制轮廓线
ax.set_zlim(-2,2)
plt.show()

```
![Axes3D绘制三维曲线图](https://i.loli.net/2019/04/14/5cb2db798d96e.jpg)


## subplot绘制多图
```python
plt.figure()

plt.subplot(2,1,1)  # 第1行只有1列
plt.plot([0,1],[0,1])

plt.subplot(2,3,4)
plt.plot([0,1],[0,1])

plt.subplot(2,3,5)
plt.plot([0,1],[0,1])

plt.subplot(2,3,6)
plt.plot([0,1],[0,1])
plt.show()


```
![subplot绘制多图](https://i.loli.net/2019/04/14/5cb2dbfda60c5.jpg)

## figure绘制多图
```python
plt.figure()

# (0,0)表示从第1行第1列开始画图，整个图形占1行3列
ax1 = plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=1)
plt.plot([0,1],[0,1])

# (1,0)表示从第2行第1列开始画图，整个图形占1行2列
ax2 = plt.subplot2grid((3,3),(1,0),colspan=2,rowspan=1)
plt.plot([0,1],[0,1])

# (1,2)表示从第2行第3列开始画图，整个图形占1行1列
ax3 = plt.subplot2grid((3,3),(1,2),colspan=1,rowspan=1)
plt.plot([0,1],[0,1])

# (2,0)表示从第3行第1列开始画图，整个图形占1行3列
ax4 = plt.subplot2grid((3,3),(2,0),colspan=3,rowspan=1)
plt.plot([0,1],[0,1])
plt.show()


```
![figure绘制多图](https://i.loli.net/2019/04/14/5cb2dc7abaf9e.jpg)

## figure图的嵌套
```python
fig = plt.figure() # 整个图形

x = [1,2,3,4,5,6,7]
y = [1,3,4,2,5,8,6]

# figure的百分比，从figure距左边10%长度，距下边10%长度的位置开始绘制，宽高是figure宽高的80%
left, bottom, width, height= 0.1, 0.1, 0.8, 0.8

# 获得绘制的句柄
ax1 = fig.add_axes([left,bottom,width,height])
# 获得绘制的句柄
ax1.plot(x,y,'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('test')

# 嵌套方法一
# figure的百分比，从figure距左边20%长度，距下边60%长度的位置开始绘制，宽高是figure宽高的25%
left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
# 获得绘制的句柄
ax2 = fig.add_axes([left,bottom,width,height])
# 绘制点(x,y)
ax2.plot(x,y,'r')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('part1')


# 嵌套方法二
left, bottom, width, height = 0.6, 0.2, 0.25, 0.25
plt.axes([left, bottom,width,height])
plt.plot(x,y,'r')
plt.xlabel('x')
plt.ylabel('y')
plt.title('part2')

plt.show()

```
![figure图的嵌套](https://i.loli.net/2019/04/14/5cb2dcdb673e3.jpg)


## 主次坐标轴
```python
x = np.arange(0,10,0.1)
y1 = 0.05*x**2
y2 = -1 * y1

fig, ax1 = plt.subplots()

# 得到ax1的对称轴ax2
ax2 = ax1.twinx()

#绘制图像
ax1.plot(x,y1,'g-')
ax2.plot(x,y2,'b--')

#设置label
ax1.set_xlabel('X data')
ax1.set_ylabel('Y1',color='g')
ax2.set_ylabel('Y2',color='b')

plt.show()


```
![主次坐标轴](https://i.loli.net/2019/04/14/5cb2de335420f.jpg)

