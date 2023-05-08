# FROM registry.access.redhat.com/ubi8/ubi
FROM rockylinux/rockylinux:8

RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm \
             https://download.fmi.fi/smartmet-open/rhel/8/x86_64/smartmet-open-release-21.3.26-2.el8.fmi.noarch.rpm

RUN dnf -y install dnf-plugins-core && \
    dnf -y module enable python38 && \
    dnf config-manager --set-enabled powertools && \
    dnf config-manager --setopt="epel.exclude=eccodes*" --save && \
    dnf -y --setopt=install_weak_deps=False install python38-pip eccodes git && \
    dnf -y clean all && rm -rf /var/cache/dnf
 # codeready-builder-for-rhel-8-x86_64-rpms && \

RUN git clone https://github.com/fmidev/snwc_bc.git

WORKDIR /snwc_bc

ADD xgb_RH_tuned23.joblib /snwc_bc
ADD xgb_WS_tuned23.joblib /snwc_bc
ADD xgb_WG_tuned23.joblib /snwc_bc
ADD xgb_T2m_tuned23.joblib /snwc_bc
ADD https://lake.fmi.fi/dem-data/DEM_100m-Int16.tif /snwc_bc
ADD requirements.txt /snwc_bc

RUN chmod 644 DEM_100m-Int16.tif && \
    update-alternatives --set python3 /usr/bin/python3.8 && \
    python3 -m pip --no-cache-dir install -r requirements.txt
