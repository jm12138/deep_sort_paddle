# **Deep Sort算法的Paddle实现**

## **简介**

* 基于[DeepSort](https://github.com/nwojke/deep_sort)官方开源代码开发，将其中的深度学习模型更换为Paddle模型

* 预训练检测模型来自[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)套件中的[特色垂类检测模型](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/featured_model/CONTRIB_cn.md)

* 兼容PaddleDetection套件导出的其他检测模型（单类别）

* 预训练特征提取模型基于Paddle官方模型库中的[Metric Learning](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning)模型开发

* 目前仅支持单类别多目标追踪

## **效果展示**

* 行人多目标追踪：

![行人多目标追踪](https://ai-studio-static-online.cdn.bcebos.com/f1225a0ec9a04fb794351efdcedc3709e9004f68f96b4f1c96d7a59cfe5f5be7)

## **快速使用**

* 同步代码：

```shell
$ git clone https://github.com/jm12138/deep_sort_paddle.git
```

* 下载预训练模型（行人多目标追踪）：[链接](http://bj.bcebos.com/v1/ai-studio-online/7e9d35a4c3f74a5b8d86220af0a082bb1b98e718ec084d149790bcf3dbb291bb?responseContentDisposition=attachment%3B%20filename%3Dmodel.zip&authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2020-11-13T07%3A01%3A51Z%2F-1%2F%2F65900c82aabcc6a3614094aeda14cdbd56ec00b63f845acbf034168b15883278)

* 预测推理：
```shell
$ cd deep_sort_paddle

$ python main.py \
    --video_path PATH_TO_VIDEO \
    --save_dir PATH_SAVE_DIR \
    --det_model_dir DET_MODEL_DIR \
    --emb_model_dir EMB_MODEL_DIR \
    --use_gpu SET_IT_IF_USE_GPU \
    --display SET_IT_IF_DISPLAY_RESULTS  
```
