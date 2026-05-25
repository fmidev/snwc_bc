FROM rockylinux/rockylinux:8

RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm \
             https://download.fmi.fi/smartmet-open/rhel/8/x86_64/smartmet-open-release-latest-8.noarch.rpm

RUN dnf -y install dnf-plugins-core && \
    dnf config-manager --set-enabled powertools && \
    dnf config-manager --setopt="epel.exclude=eccodes*" --save && \
    dnf -y --setopt=install_weak_deps=False install python3.11 python3.11-pip python3.11-setuptools eccodes git && \
    dnf -y clean all && rm -rf /var/cache/dnf

ADD . /snwc_bc

WORKDIR /snwc_bc

ENV MODEL_VERSION 15km_0526
ARG S3_HOSTNAME=lake.fmi.fi

ADD https://${S3_HOSTNAME}/dem-data/DEM_100m-Int16.tif /snwc_bc
ADD https://${S3_HOSTNAME}/ml-models/mnwc-biascorrection/xgb_T2m_${MODEL_VERSION}.joblib /snwc_bc
ADD https://${S3_HOSTNAME}/ml-models/mnwc-biascorrection/xgb_WS_${MODEL_VERSION}.joblib /snwc_bc
ADD https://${S3_HOSTNAME}/ml-models/mnwc-biascorrection/xgb_WG_${MODEL_VERSION}.joblib /snwc_bc
ADD https://${S3_HOSTNAME}/ml-models/mnwc-biascorrection/xgb_RH_${MODEL_VERSION}.joblib /snwc_bc

RUN chmod 644 DEM_100m-Int16.tif && \
    chmod 644 xgb_T2m_${MODEL_VERSION}.joblib && \
    chmod 644 xgb_WS_${MODEL_VERSION}.joblib && \
    chmod 644 xgb_WG_${MODEL_VERSION}.joblib && \
    chmod 644 xgb_RH_${MODEL_VERSION}.joblib && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    python3 -m pip --no-cache-dir install -r requirements.txt
