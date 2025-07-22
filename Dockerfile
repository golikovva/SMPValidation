FROM pytorch/pytorch

RUN pip install scikit-learn
RUN pip install pandas
RUN pip install netCDF4
RUN pip install matplotlib
RUN pip install pendulum
RUN conda install -c conda-forge wrf-python=1.3.4.1
RUN pip install transformers
RUN pip install SciPy
RUN pip install optuna
RUN pip install jupyter
RUN pip install jupyterlab
RUN pip install notebook
RUN pip install addict
# RUN pip install opencv-python-headless
# RUN pip install basemap
RUN pip install pytorch-msssim
RUN conda update -n base -c defaults conda
RUN conda install -c conda-forge esmpy
RUN pip install pyproj
RUN pip install global-land-mask
RUN pip install cartopy
RUN pip install pygrib
RUN pip install geopandas
RUN pip install rasterio
RUN pip install cmocean



EXPOSE 9999
ENV NAME vgolikov_validation
COPY . /home

WORKDIR /home/experiments/train_test

