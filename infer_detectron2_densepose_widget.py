from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_detectron2_densepose.infer_detectron2_densepose_process import InferDetectron2DenseposeParam
# PyQt GUI framework
from PyQt5.QtWidgets import *
import torch


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferDetectron2DenseposeWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferDetectron2DenseposeParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

        self.check_cuda = pyqtutils.append_check(self.gridLayout, "Cuda", self.parameters.cuda and
                                                 torch.cuda.is_available())
        self.check_cuda.setEnabled(torch.cuda.is_available())
        self.spin_thr = pyqtutils.append_double_spin(self.gridLayout, "Detection threshold", self.parameters.conf_thres, min=0,
                                                     max=1, step=0.01, decimals=2)
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        # Example : self.parameters.windowSize = self.spinWindowSize.value()
        self.parameters.conf_thres = self.spin_thr.value()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.update = True
        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferDetectron2DenseposeWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_detectron2_densepose"

    def create(self, param):
        # Create widget object
        return InferDetectron2DenseposeWidget(param, None)
