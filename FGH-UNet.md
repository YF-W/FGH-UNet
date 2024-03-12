# A U-shaped Network for Fine-Grained Medical Image Processing Based on HorBlock

***In the realm of medical image processing, HorBlock undergoes Depth-Wise Convolution (DWConv) at each stage to ensure long-term dependency. However, it exhibits limitations in handling image details, especially with more intricate medical images. Unlike other scenarios, the challenges posed by medical images include the irregular edges of lesions, their blending with normal tissue colors, and the presence of irregular texture trends. Addressing these challenges requires the model to possess enhanced feature extraction and retention capabilities, effective noise reduction and smoothing effects, and a more powerful network expression. Therefore, effectively leveraging the advantages of HorBlock in medical image processing remains a complex and challenging task.***

# Paper:FGH-UNet（A U-shaped Network for Fine-Grained Medical Image Processing Based on HorBlock）

**Authors : Yuefei Wang ,Li Zhang, Yutong Zhang, Haoyue Cai,  Yuquan Xu, Ronghui Feng, Binxiong Li, Xue Li,Yixi ,Xu Xiang, Xi Yu**

## **1.Architecture Overview (模型架构概述)**

![img](file:///C:\Users\lp2e\Documents\Tencent Files\1066059986\Image\C2C\2416BB7E9A9EE59697FE4FC267F90C32.png)

***FGH-UNet is a "single encoder - single decoder" network structure. The encoder focuses on image feature extraction, using a residual network structure to prevent learning degradation. To enhance semantic information transmission between encoding and decoding, we introduce the BiAPyra HorBlock with global attention and a bidirectional pyramid module for image detail processing. This approach improves feature learning and addresses learning degradation, especially in complex medical images with irregular lesions and textures. Additionally, integrated attention analysis at the intersection of encoding and decoding helps allocate weights to feature images, improving the model's ability to focus on sensitive and meaningful regions during image upsampling.***

## 2.Module 1：BiAPyra HorBlock

![image-20240312150400823](C:\Users\lp2e\AppData\Roaming\Typora\typora-user-images\image-20240312150400823.png)

***To enhance the advantages of HorBlock in handling image details, the BiAPyra HorBlock introduces a Bidirectional Asymmetric Pyramid model. This network retains a multi-stage architecture and emphasizes extracting image details after processing features with large kernels (7*7) at each stage. In semantic segmentation research, various methods exist for extracting texture details, and the multi-branch processing approach often achieves a balanced and comprehensive analysis of pixel classification through multiple methods.***

## 3.Module 2:   Reinforcement of attention mechanisms

![image-20240312150445610](C:\Users\lp2e\AppData\Roaming\Typora\typora-user-images\image-20240312150445610.png)

***At the intersection of encoding and decoding, the bottleneck region is filled with rich feature tensors, making it a crucial area in the entire network. The PS-CJ module can simultaneously consider both global context and local details, enhancing the understanding of image semantics. This module is designed to fully learn, absorb, and refine all key features of the network.***

# **Datasets**:

1.The LUNG dataset:https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data.

2.The Skin Lesions dataset :https://www.kaggle.com/datasets/ojaswipandey/skin-lesion-dataset.

3.The DRIVE dataset:https://drive.grand-challenge.org/.

4.The MICCAI2015-CVC ClinicDB dataset: https://polyp.grands-challenge.org/CVCClinicDB/.

5.The TNSCUI2020 - Thyroid Nodule dataset: https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st.