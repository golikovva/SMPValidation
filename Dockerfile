FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

SHELL ["bash", "-lc"]

# (опционально, но полезно) строгий приоритет каналов, чтобы меньше было "солянки" бинарников
RUN conda config --set channel_priority strict

RUN conda update -n base -c defaults -y conda \
 && conda install -n base -c conda-forge -y mamba \
 && mamba install -n base -c conda-forge -y \
      wrf-python \
      esmpy \
      opendrift \
 && mamba clean -a -y

RUN pip install --no-cache-dir \
    scikit-learn pandas netCDF4 matplotlib pendulum transformers scipy optuna \
    jupyter jupyterlab notebook addict pytorch-msssim \
    pyproj global-land-mask cartopy pygrib geopandas rasterio cmocean

# Смоук-тест: важны именно from_numpy / numpy()
RUN python -c "import numpy as np, torch; print('numpy', np.__version__); print('torch', torch.__version__); print(torch.from_numpy(np.array([1],dtype=np.int64)))"
# RUN python -c "import opendrift; print('OpenDrift', opendrift.__version__)"

EXPOSE 9999
ENV NAME vgolikov_validation
COPY . /home
WORKDIR /home/experiments/train_test
