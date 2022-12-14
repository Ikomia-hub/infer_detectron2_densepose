from infer_detectron2_densepose import update_path
from ikomia import core, dataprocess
import copy
from infer_detectron2_densepose.densepose.structures import DensePoseChartPredictorOutput, \
    DensePoseEmbeddingPredictorOutput
from infer_detectron2_densepose.densepose import add_densepose_config
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from infer_detectron2_densepose.densepose.vis.extractor import (
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
)


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferDetectron2DenseposeParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        # Example : self.windowSize = 25
        self.cuda = torch.cuda.is_available()
        self.thr = 0.8

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(param_map["windowSize"])
        self.cuda = eval(param_map["cuda"])
        self.thr = float(param_map["thr"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        # Example : paramMap["windowSize"] = str(self.windowSize)
        param_map["cuda"] = str(self.cuda)
        param_map["thr"] = str(self.thr)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferDetectron2Densepose(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        # Example :  self.addInput(dataprocess.CImageIO())
        #           self.addOutput(dataprocess.CImageIO())
        self.addOutput(dataprocess.CGraphicsOutput())
        # Create parameters class
        if param is None:
            self.setParam(InferDetectron2DenseposeParam())
        else:
            self.setParam(copy.deepcopy(param))
        self.cfg = get_cfg()
        add_densepose_config(self.cfg)
        self.cfg.merge_from_file(os.path.dirname(__file__) + "/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
        self.cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
        self.predictor = None
        self.thr = 0.8

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()
        # Get input
        input_img = self.getInput(0)
        img = input_img.getImage()

        # Get graphic output
        graphics_output = self.getOutput(1)
        graphics_output.setNewLayer("Densepose")
        graphics_output.setImageIndex(0)
        self.forwardInputImage(0, 0)

        levels = np.linspace(0, 1, 9)
        prop_line = core.GraphicsPolylineProperty()
        prop_line.line_size = 1
        param = self.getParam()
        
        if self.predictor is None or param.update:
            self.cfg.MODEL.DEVICE = 'cuda' if param.cuda else 'cpu'
            self.predictor = DefaultPredictor(self.cfg)
            self.thr = param.thr
            param.update = False
            
        with torch.no_grad():
            outputs = self.predictor(img)["instances"]
            scores = outputs.get("scores").cpu()
            if outputs.has("pred_boxes"):
                pred_boxes_XYXY = outputs.get("pred_boxes").tensor.cpu()
                if outputs.has("pred_densepose"):
                    if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                        extractor = DensePoseResultExtractor()
                    elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                        extractor = DensePoseOutputsExtractor()
                    pred_densepose = extractor(outputs)[0]
                    for i, (score, truc, box_xyxy) in enumerate(zip(scores, pred_densepose, pred_boxes_XYXY)):
                        if score > self.thr:
                            x1, y1, x2, y2 = box_xyxy
                            u = truc.uv[0, :, :].cpu().numpy()
                            v = truc.uv[1, :, :].cpu().numpy()

                            iso_u = plt.contour(u, levels, extent=(x1, x2, y1, y2))
                            iso_v = plt.contour(v, levels, extent=(x1, x2, y1, y2))

                            self.visualize(iso_u.collections, graphics_output, prop_line)
                            self.visualize(iso_v.collections, graphics_output, prop_line)

                            graphics_output.addRectangle(float(x1), float(y1), float(x2 - x1),
                                                         float(y2 - y1))

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def visualize(self, collections, graphics_output, properties_line):
        for i in range(len(collections)):
            color = collections[i].get_colors()[0]
            properties_line.pen_color = [int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)]
            for lst_pts in collections[i].get_segments():
                pts = []
                for j in range(len(lst_pts)):
                    pts.append(core.CPointF(float(lst_pts[j][0]),
                                            float(lst_pts[j][1])))
                graphics_output.addPolyline(pts, properties_line)


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferDetectron2DenseposeFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_detectron2_densepose"
        self.info.shortDescription = "Detectron2 inference model for human pose detection."
        self.info.description = "Inference model for human pose detection trained on COCO dataset. " \
                                "Implementation from Detectron2 (Facebook Research). " \
                                "Dense human pose estimation aims at mapping all human pixels " \
                                "of an RGB image to the 3D surface of the human body. " \
                                "The model used is composed by ResNet50 backbone + panoptic FPN head."
        self.info.authors = "R??za Alp G??ler, Natalia Neverova, Iasonas Kokkinos"
        self.info.article = "DensePose: Dense Human Pose Estimation In The Wild"
        self.info.journal = "Conference on Computer Vision and Pattern Recognition (CVPR)"
        self.info.year = 2018
        self.info.license = "Apache-2.0 License"
        self.info.version = "1.1.2"
        self.info.repo = "https://github.com/facebookresearch/detectron2/tree/master/projects/DensePose"
        self.info.documentationLink = "https://detectron2.readthedocs.io/index.html"
        self.info.path = "Plugins/Python/Pose"
        self.info.iconPath = "icons/detectron2.png"
        self.info.keywords = "human,pose,detection,keypoint,facebook,detectron2,mesh,3D surface"

    def create(self, param=None):
        # Create process object
        return InferDetectron2Densepose(self.info.name, param)
