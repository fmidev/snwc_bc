FROM rockylinux/rockylinux:8

RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm \
             https://download.fmi.fi/smartmet-open/rhel/8/x86_64/smartmet-open-release-latest-8.noarch.rpm

RUN dnf -y install dnf-plugins-core && \
    dnf config-manager --set-enabled powertools && \
    dnf config-manager --setopt="epel.exclude=eccodes*" --save && \
    dnf -y --setopt=install_weak_deps=False install python3.11 python3.11-pip python3.11-setuptools eccodes git && \
    dnf -y clean all && rm -rf /var/cache/dnf

RUN git clone https://github.com/fmidev/snwc_bc.git

WORKDIR /snwc_bc

ADD https://lake.fmi.fi/dem-data/DEM_100m-Int16.tif /snwc_bc
ADD https://lake.fmi.fi/ml-models/mnwc-biascorrection/xgb_T2m_1023.joblib /snwc_bc
ADD https://lake.fmi.fi/ml-models/mnwc-biascorrection/xgb_WS_1023.joblib /snwc_bc
ADD https://lake.fmi.fi/ml-models/mnwc-biascorrection/xgb_WG_1023.joblib /snwc_bc
ADD https://lake.fmi.fi/ml-models/mnwc-biascorrection/xgb_RH_1023.joblib /snwc_bc

RUN chmod 644 DEM_100m-Int16.tif && \
    chmod 644 xgb_T2m_1023.joblib && \
    chmod 644 xgb_WS_1023.joblib && \
    chmod 644 xgb_WG_1023.joblib && \
    chmod 644 xgb_RH_1023.joblib && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    python3 -m pip --no-cache-dir install -r requirements.txt
