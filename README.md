### GRU

2017年12月，在dl4j基础上实现一些算法，现根据GRU前向公式推导反向公式，并在dl4j中实现。
<br />
<br />

### Gan示例

结构：
<br />
![image](https://user-images.githubusercontent.com/35036729/115952807-c05d3a00-a51a-11eb-9555-05aa68e0e2ce.png)

**1、Generator和Discriminator使用Dense**
<br />
https://github.com/Gerry-Pan/pan-dl4j/blob/master/src/main/java/personal/pan/dl4j/examples/gan/GanTrainer.java
<br />
<br />
训练效果：
<br />
![输入图片说明](https://images.gitee.com/uploads/images/2021/0422/084047_e42588ce_673907.png "QQ图片20210422083446.png")
![输入图片说明](https://images.gitee.com/uploads/images/2021/0422/084313_23216ff4_673907.png "QQ图片20210422083521.png")

**2、Generator使用上采样和Cnn，Discriminator使用Cnn**
<br />
https://github.com/Gerry-Pan/pan-dl4j/blob/master/src/main/java/personal/pan/dl4j/examples/gan/ConvGanTrainer.java
<br />
<br />
训练效果：
<br />
![image](https://user-images.githubusercontent.com/35036729/115952776-94da4f80-a51a-11eb-8ab6-87360f25720f.png)
<br />
<br />
进一步调优训练结果，笔画更清晰：
<br />
![QQ图片20210518071401](https://user-images.githubusercontent.com/35036729/118568156-cbb62500-b7a9-11eb-85a7-334477f64456.png)
![QQ图片20210518071407](https://user-images.githubusercontent.com/35036729/118568159-cce75200-b7a9-11eb-9c3e-4e85ce0ce89a.png)

### CGan示例
**1、Generator使用上采样和Cnn，Discriminator使用Cnn**
<br />
https://github.com/Gerry-Pan/pan-dl4j/blob/master/src/main/java/personal/pan/dl4j/examples/gan/ConditionalConvGanTrainer.java
<br />
<br />
训练效果：
<br />
![QQ图片20210518224427](https://user-images.githubusercontent.com/35036729/118673200-673baa00-b82b-11eb-9cc3-031b1e23a493.png)
![QQ图片20210518224430](https://user-images.githubusercontent.com/35036729/118673218-69056d80-b82b-11eb-97db-0f5098e4f8ce.png)
![QQ图片20210518224436](https://user-images.githubusercontent.com/35036729/118673222-699e0400-b82b-11eb-9fd3-76cdc358cd21.png)




