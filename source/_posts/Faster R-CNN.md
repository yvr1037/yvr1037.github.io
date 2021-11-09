---

title: Faster-RCNN
data: 2021-10-27 
tags:
  - [cv]
  - [RCNN]

---



##### Perface:

在🦌同学的感染下，笔者最近也学习了目标检测方向的相关内容，看的第一篇论文是[Faster R-CNN：Towards Rel-Time Objection Dection with Region Proposal Networks](https://arxiv.org/abs/1504.08083#)，里面涉及到很多前置模型需要了解结构，在这里分享一点笔记

##### 目标检测背景

目标检测是很多计算机视觉人物的基础，目前主流的目标检测的算法主要基于深度学习模型可以分为两大类

1. one-stage检测算法,这种算法直接产生物体的类别概率和坐标位置,不需要直接产生候选区域.比如说YOLO和SSD
2. two-stage检测算法,这是将检测问题划分为两个阶段,首先是产生候选区域,然后对候选区域分类;典型算法是R-CNN系列,faster rcnn就是基于**region proposal**(候选区域)

##### backbone network

**Faster R-CNN**使用的主干网络是VGG-16,在论文中称主干网络时**backbone network**,主干网络就是用来**feature extraction**,当然这个不是一成不变的,可以替换,比如现在也同样流行使用**Resnet**,再如**CornerNet**算法中使用的backbone network是Hourglass Network.
关于VGG-16可以参考[VGG介绍](http://zh.gluon.ai/chapter_convolutional-neural-networks/vgg.html),16的含义是含有参数有16层,分别是13个卷积层+3个全连接层

<img src="C:\Users\VrShadow\AppData\Roaming\Typora\typora-user-images\image-20211027195132710.png" alt="image-20211027195132710" style="zoom:65%;" />

图来自网络

<img src="C:\Users\VrShadow\AppData\Roaming\Typora\typora-user-images\image-20211027195339689.png" alt="image-20211027195339689" style="zoom:60%;" />

##### Faster R-CNN算法步骤

这部分是为了理解Faster R-CNN,总体描述下算法的整个过程以便后期做细节分析

<img src="C:\Users\VrShadow\AppData\Roaming\Typora\typora-user-images\image-20211027195754031.png" alt="image-20211027195754031" style="zoom:50%;" />

大致流程是:将整张图片输入CNN层,得到feature map,卷积特征输入到**RPN(Region Proposal Network)**得到候选框的特征信息,对候选框中提取的特征使用分类器判别是否属于一个特定类别,对于属于某一特征的候选框用回归器进一步调整其位置.

<img src="C:\Users\VrShadow\AppData\Roaming\Typora\typora-user-images\image-20211027200749657.png" alt="image-20211027200749657" style="zoom:50%;" />

Faster R-CNN可以看作RPN和Fast R-CNN模型的结合,即Faster R-CNN = RPN + Fast R-CNN.下面介绍每一步骤的输入输出的细节.

+ 首先通过预训练模型训练得到Conv layers(这个conv layer实际上就是VGG-16)能够接收整张图片并提取特征图feature maps,这个feature map是在conv层之后获得的特征.
+ feature map被共享之后用于后续的RPN和Rol池化层
  - BPN层:BPN网络用于生成region proposals.该层通过softmax判断anchors属于前景(foreground)还是背景(background),再利用边框回归修正anchors,获得精确的proposals 
  - RoI Pooling层:该层收集输入的feature map和proposals综合这些信息提取proposal feature map,进入到后面可利用全连接操作层进行目标识别和定位
+ 最后的classifier会将Roi Pooling层形成固定大小的feature map进行全连接操作,利用softmax进行具体类别的分类,同时利用L1 loss完成bounding box regression回归操作获得物体的准确位置

##### 细节

###### 1.RPN

之前的R-CNN和Fast R-CNN都是采用可选择性搜索(SS)来产生候选框的,但是这种方法特别耗时;Faster R-CNN最大的亮点是抛弃SS,采用RPN生成候选框.
<img src="C:\Users\VrShadow\AppData\Roaming\Typora\typora-user-images\image-20211027204057233.png" alt="image-20211027204057233" style="zoom:67%;" />

说明:

1. Conv feature map:VGG-16网络最后一个卷积层输出的feature map
2. Sliding window:滑动窗口实际上就是3*3的卷积核,滑窗只要选取所有可能的区域并没有额外的作用
3. K anchor boxes:在每个sliding window的点上初始化的参考区域(论文中k=9)就是9个矩形框
4. Intermediate layer:中间层，256-d是中间层的维度(论文中谁用ZF网络就是256维,VGG就是512维)
5. Cls layer:分类层,预测proposal的anchor对应的proposal的(x,y,w,h)
6. 2k scores:2k个分数(18个)
7. Reg layer:回归层,判断该proposal是前景还是背景
8. 4k coordinates:4k坐标(36个)

- RPN的输入是卷积特征图,输出是图片生成的proposals,RPN通过一个滑动窗口连接在最后一个卷积层的feature map上,生成一个长度256的全连接特征
- 这个全连接层特征分别送入两个全连接层一个是分类层,用于分类检测;一个是回归层,用于回归;对于每个滑动窗口位置一般设置k(论文中k=9)个不同大小或者比例的anchors这意味着每个滑窗覆盖的位置就会预测9哥候选区域
  **分类层**:每个anchor输出两个预测值:anchor是背景(background,非object)的score和anchor是前景(foreground,object)的score
  **回归层**:输出4k(4*9=36)个坐标值表示每个候选区域的位置(x,y,w,h)

也就是说我么是通过这些特征图应用滑动窗口加anchor机制进行目标区域判定和分类的,这里的滑窗加anchor机制功能类似于fast rcnn的selective search生成proposals的作用,而我们是通过RPN生成proposals.RPN就是一个卷积层 + relu +左右两个层(cls layer和reg layer)的小型网络

###### 2.anchor

论文内容:The k proposals are parameterized relative to k reference boxes, which we call anchors;可以理解为锚点位于之前说的3 * 3的滑窗中心处,就是因为有多个anchor.这9个anchor是作者设置的,论文中scale=[128,256,512],长宽比[1:1,1:2,2:1]有9种；自己可以根据目标的特点做出不同的设计;对于一幅 w * h的feature map一共有w * h * k个锚点.

<img src="C:\Users\VrShadow\AppData\Roaming\Typora\typora-user-images\image-20211027213420072.png" alt="image-20211027213420072" style="zoom:50%;" />

###### 3.VGG提取特征

VGG的网络流程图:

<img src="C:\Users\VrShadow\AppData\Roaming\Typora\typora-user-images\image-20211027213725540.png" alt="image-20211027213725540" style="zoom:67%;" />

每个卷积层利用前面网络信息生成抽象描述:
第一层学习边缘edges信息；
第二层:学习边缘edges中图案patterns以学习更加复杂的形状信息；最终得到卷积特征图其空间维度(分辨率)比原图小了很多但更深；
特征图的width和height由于卷积层间的池化层而降低,而depth由于卷积层学习的filters数量而增加.

###### 4.ROI pooling

ROI就是region of interest指的是感兴趣区域;如果是原图，roi就是目标，如果是featuremap，roi就是特征图像目标的特征了，roi在这里就是经过RPN网络得到的，总之就是一个框。pooling就是池化。所以ROI Pooling就是Pooling的一种，只是是针对于Rois的pooling操作而已。RPN 处理后，可以得到一堆没有 class score 的 object proposals.待处理问题为：如何利用这些proposals分类.Roi pooling层的过程就是为了将不同输入尺寸的feature map（ROI）抠出来，然后resize到统一的大小.

ROI pooling层的输入:

1. 特征图features map(这个特征图就是cnn卷积出来以后用于共享的那个特征图)
2. roi信息:(就是RPN网络的输出,一个表示所有ROI的N*5矩阵,N表示ROI的数目;第一列表示图像index,其余四列表示其余的左上角和右下角坐标,坐标信息是对应原图中的绝对坐标)

ROI pooling层的过程:

首先将RPN中得到的原图中roi信息映射到feature map上按原图与featuremap的比例缩小roi坐标就行了），然后经过最大池化，池化到固定大小w×h。但这个pooling不是一般的Pooling，而是将区域等分，然后取每一小块的最大值，最后才能得到固定尺寸的roi。

也就是：

根据输入的image，将Roi映射到feature map对应的位置；
将映射后的区域划分为相同大小的sections（sections数量和输出的维度相同）；
对每个section进行max pooling操作；
ROI pooling层的输出：

结果是，由一组大小各异的矩形，我们快速获取到具有固定大小的相应特征图。值得注意的是，RoI pooling 输出的维度实际上并不取决于输入特征图的大小，也不取决于区域提案的大小。这完全取决于我们将区域分成几部分。也就是，batch个roi矩阵，每一个roi矩阵为：通道数xWxH,也就是从selective search得到batch个roi，然后映射为固定大小。

###### 5.NMS

NMS（Non Maximum Suppression，非极大值抑制）用于后期的物体冗余边界框去除，因为目标检测最终一个目标只需要一个框，所以要把多余的框干掉，留下最准确的那个。

NMS的输入：

检测到的Boxes(同一个物体可能被检测到很多Boxes，每个box均有分类score)

NMS的输出：

最优的Box.

###### 6.FC layer

经过roi pooling层之后，batch_size=300, proposal feature map的大小是7×7,512-d,对特征图进行全连接，参照下图，最后同样利用Softmax Loss和L1 Loss完成分类和定位。

![image-20211027220232527](C:\Users\VrShadow\AppData\Roaming\Typora\typora-user-images\image-20211027220232527.png)

通过全连接层与softmax计算每个region proposal具体属于哪个类别（如人，马，车等），输出cls_prob概率向量；同时再次利用bounding box regression获得每个region proposal的位置偏移量bbox_pred，用于回归获得更加精确的目标检测框

即从PoI Pooling获取到7x7大小的proposal feature maps后，通过全连接主要做了：

通过全连接和softmax对region proposals进行具体类别的分类；

再次对region proposals进行bounding box regression，获取更高精度的rectangle box。

##### 主要部分

**Faster** **RCNN**其实可以分为四部分主要内容

###### 1.Conv Layer

作为一种CNN目标检测方法,Faster RCNN首先使用一组基础的cnn+relu+pooling层提取image的feature map,这个feature map被共享用用于后续RPN层和全连接层

###### 2.Region Proposal NetWorks

RPN网络用于生成region proposals,该层通过softmax判断anchors属于positive还是negative,再利用bounding

box regression修正anchors获得精确的proposals

###### 3.Roi Pooling

该层手机输入的feature map和proposals,综合这些信息之后提取proposals,综合这些信息提取proposals feature maps送入后续全连接层判定目标类别

###### 4.Classfication

利用proposals feature map计算proposals的类别同时再次bounding box regression获得检测框最终的精确位置

<img src="C:\Users\VrShadow\AppData\Roaming\Typora\typora-user-images\image-20211028212022996.png" alt="image-20211028212022996" style="zoom:67%;" />

上图展示了python版本中的VGG16模型中的faster rcnn的网络结构可以清晰的看到该网络对于一幅任意大小的P*Q的图像:

- 首先固定至大小M×N然后将M×N图像送入网络;
- 而Conv layer