from libs.validation.datasets.amsr_jaxa import (
    AmsrJaxaHsi10kmDataset,
    AmsrJaxaHsi25kmDataset,
    AmsrJaxaSimRDataset,
    AmsrJaxaSimYDataset,
)
from libs.validation.datasets.cryosat import (
    CryosatThickDataset,
)
from libs.validation.datasets.glorys import (
    GlorysOperativeCorrectedSalinityDataset,
    GlorysOperativeCurrentVelocityDataset,
    GlorysOperativeDriftDataset,
    GlorysOperativeEastCurrentDataset,
    GlorysOperativeNorthCurrentDataset,
    GlorysOperativeSalinityDataset,
    GlorysOperativeSicDataset,
    GlorysOperativeTemperatureDataset,
    GlorysOperativeThickDataset,
    GlorysReanalysisCurrentVelocityDataset,
    GlorysReanalysisDriftDataset,
    GlorysReanalysisEastCurrentDataset,
    GlorysReanalysisNorthCurrentDataset,
    GlorysReanalysisSalinityDataset,
    GlorysReanalysisSicDataset,
    GlorysReanalysisTemperatureDataset,
    GlorysReanalysisThickDataset,
)
from libs.validation.datasets.nemo import (
    NemoDriftDataset,
    NemoEastCurrentDataset,
    NemoNorthCurrentDataset,
    NemoSalinityDataset,
    NemoSicDataset,
    NemoTemperatureDataset,
    NemoThickDataset,
)
from libs.validation.datasets.shapefile import (
    ShapefileDriftDataset,
    ShapefileSicDataset,
    ShapefileThickDataset,
)
from libs.validation.datasets.stubs import (
    ConstantDataset,
    FusionDataset,
)
