# **Deep Sort 算法的 Paddle 实现**
## 更新
* 支持 PaddleDetection 2.x 动态图版本导出的检测模型
* 支持追踪结果以文本的形式进行输出保存

## **简介**

* 基于 [DeepSort](https://github.com/nwojke/deep_sort) 官方开源代码开发，将其中的深度学习模型更换为 Paddle 模型

* 预训练检测模型来自 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) 套件中的[特色垂类检测模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1/configs/pedestrian)

* 兼容PaddleDetection套件导出的其他检测模型（单类别）

* 预训练特征提取模型基 于Paddle 官方模型库中的 [Metric Learning](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning) 模型开发

* 目前仅支持单类别多目标追踪

## **效果展示**

* 行人多目标追踪：

    ![行人多目标追踪](https://ai-studio-static-online.cdn.bcebos.com/f1225a0ec9a04fb794351efdcedc3709e9004f68f96b4f1c96d7a59cfe5f5be7)

## **快速使用**

* 同步代码：

    ```shell
    $ git clone https://github.com/jm12138/deep_sort_paddle.git
    ```

* 下载预训练模型（行人多目标追踪）：[链接](https://bj.bcebos.com/v1/ai-studio-online/ce746f80dd7b4f329efeb076c75d64c858f5f065900548a18e1dc58cd3b981b5?responseContentDisposition=attachment%3B%20filename%3Dmodel_ppdet_2.x.zip)

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

    * 更多参数请查看 [main.py](./main.py) 源代码
