import os
import cv2
import numpy as np
from paddle.inference import Config
from paddle.inference import create_predictor

__all__ = ['Embedding']

class Embedding():
    def __init__(self, model_dir, use_gpu=False):
        self.predictor, self.input_handle, self.output_handle = self.load_model(model_dir, use_gpu)

    def load_model(self, model_dir, use_gpu=False):
        model = os.path.join(model_dir, '__model__')
        params = os.path.join(model_dir, '__params__')
        config = Config(model, params)

        # 设置参数
        if use_gpu:   
            config.enable_use_gpu(100, 0)
        else:
            config.disable_gpu()
            config.enable_mkldnn()
        config.disable_glog_info()
        config.switch_ir_optim(True)
        config.enable_memory_optim()
        config.switch_use_feed_fetch_ops(False)
        config.switch_specify_input_names(True)

        # 通过参数加载模型预测器
        predictor = create_predictor(config)

        # 获取模型的输入输出
        input_names = predictor.get_input_names()
        output_names = predictor.get_output_names()
        input_handle = predictor.get_input_handle(input_names[0])
        output_handle = predictor.get_output_handle(output_names[0])

        return predictor, input_handle, output_handle

    def preprocess(self, imgs):
        im_batch = []
        for img in imgs:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img = cv2.resize(img, (128, 64))
            img = cv2.resize(img, (128, 128))
            img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
            img_mean = np.array(mean).reshape((3, 1, 1))
            img_std = np.array(std).reshape((3, 1, 1))
            img -= img_mean
            img /= img_std
            img = np.expand_dims(img, axis=0)
            im_batch.append(img)
        
        im_batch = np.concatenate(im_batch, 0)
        return im_batch

    def predict(self, imgs):
        input_datas = self.preprocess(imgs)
        self.input_handle.copy_from_cpu(input_datas)
        self.predictor.run()
        result = self.output_handle.copy_to_cpu()
        return result

if __name__ == '__main__':

    emb = Embedding('./embedding', use_gpu=True)

    imgs = [cv2.imread('new.jpg')]
    result = emb.predict(imgs)

    print(len(result[0]))
