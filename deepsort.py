import numpy as np
from model import Detector, Embedding
from deep_sort import NearestNeighborDistanceMetric, Detection, Tracker

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(
        self,
        det_model_dir,
        emb_model_dir,
        use_gpu=False,
        run_mode='fluid',
        use_dynamic_shape=False,
        trt_min_shape=1,
        trt_max_shape=1280,
        trt_opt_shape=640,
        trt_calib_mode=False,
        cpu_threads=1,
        enable_mkldnn=False,
        threshold=0.5,
        max_cosine_distance=0.2,
        nn_budget=100,
        max_iou_distance=0.9,
        max_age=70,
        n_init=3
    ):
        self.detector = Detector(
            model_dir=det_model_dir,
            use_gpu=use_gpu,
            run_mode=run_mode,
            use_dynamic_shape=use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn
        )
        self.emb = Embedding(emb_model_dir, use_gpu)
        self.threshold = threshold
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, ori_img):
        self.height, self.width = ori_img.shape[:2]
        
        results = self.detector.predict(ori_img[np.newaxis, ...], self.threshold)
        if results is None:
            return None
        else:
            tlwh, xyxy, confidences = results
            if not confidences.tolist():
                return None
        # generate detections
        features = self.get_features(xyxy, ori_img)
        detections = [Detection(tlwh[i], conf, features[i])
                      for i, conf in enumerate(confidences)]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlbr()
            x1, y1, x2, y2 = box
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def get_features(self, xyxy, ori_img):
        crops = []
        for bbox in xyxy:
            crop = ori_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            crops.append(crop)
        features = self.emb.predict(crops)
        return features
