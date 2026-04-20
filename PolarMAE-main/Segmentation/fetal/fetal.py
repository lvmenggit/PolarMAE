from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class FetalHeadBiometryDataset(BaseSegDataset):
    """Fetal 2-class dataset.

    Original mask convention:
    - 0: background
    - 1: AC
    - 2: other (HC/Fetal Head/Trans-cerebellum/Trans-thalamic/Trans-ventricular)

    With reduce_zero_label=True:
    - 0(background) -> ignore
    - 1(AC) -> 0
    - 2(other) -> 1
    """

    METAINFO = dict(
        classes=('ac', 'other'),
        palette=[
            [255, 0, 0],   # ac
            [0, 255, 0],   # other
        ]
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
