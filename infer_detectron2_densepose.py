from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate process object
        from infer_detectron2_densepose.infer_detectron2_densepose_process import InferDetectron2DenseposeFactory
        return InferDetectron2DenseposeFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_detectron2_densepose.infer_detectron2_densepose_widget import InferDetectron2DenseposeWidgetFactory
        return InferDetectron2DenseposeWidgetFactory()
