########################################################
# The FUSE driver needs elevated privileges, run Docker with --privileged=true 
# or with minimum elevation as shown below:
# $ sudo docker run -d --rm --name s3fs --security-opt apparmor:unconfined \
#  --cap-add mknod --cap-add sys_admin --device=/dev/fuse \
#  -e S3_BUCKET=MY_S3_BUCKET -e S3_REGION=ap-southeast-2 \
#  -e MNT_POINT=/data git.encompasshost.com:5001/encompass/images/s3fs:latest
########################################################
 
FROM ubuntu:14.04
 
 
ENV DUMB_INIT_VER 1.2.0
ENV S3_BUCKET mlops-platform-training-data
ENV MNT_POINT /data
# ARG S3_REGION=$AWS_REGION
# ARG AWS_KEY=$AWS_KEY 
# ARG AWS_SECRET_KEY=$AWS_SECRET_KEY 
RUN --mount=type=secret,id=AWS_REGION \
--mount=type=secret,id=AWS_KEY \
--mount=type=secret,id=AWS_SECRET_KEY \
export AWS_REGION=$(cat /run/secrets/AWS_REGION) && \
export AWS_KEY=$(cat /run/secrets/AWS_KEY) && \
export AWS_SECRET_KEY=$(cat /run/secrets/AWS_SECRET_KEY) && \
yarn gen


RUN DEBIAN_FRONTEND=noninteractive apt-get -y update --fix-missing && \
    apt-get install -y automake autotools-dev g++ git libcurl4-gnutls-dev wget \
                       libfuse-dev libssl-dev libxml2-dev make pkg-config && \
    git clone https://github.com/s3fs-fuse/s3fs-fuse.git /tmp/s3fs-fuse && \
    cd /tmp/s3fs-fuse && ./autogen.sh && ./configure && make && make install && \
    ldconfig && /usr/local/bin/s3fs --version && \
    wget -O /tmp/dumb-init_${DUMB_INIT_VER}_amd64.deb https://github.com/Yelp/dumb-init/releases/download/v${DUMB_INIT_VER}/dumb-init_${DUMB_INIT_VER}_amd64.deb && \
    dpkg -i /tmp/dumb-init_*.deb
 
RUN echo "${AWS_KEY}:${AWS_SECRET_KEY}" > /etc/passwd-s3fs && \
    chmod 0400 /etc/passwd-s3fs
 
RUN mkdir -p "$MNT_POINT"
 
RUN DEBIAN_FRONTEND=noninteractive apt-get purge -y wget automake autotools-dev g++ git make && \
    apt-get -y autoremove --purge && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
 
# Runs "/usr/bin/dumb-init -- CMD_COMMAND_HERE"
#ENTRYPOINT ["/usr/bin/dumb-init", "--"]
 
CMD exec /usr/local/bin/s3fs $S3_BUCKET $MNT_POINT -f -o endpoint=${S3_REGION},allow_other,use_cache=/tmp,max_stat_cache_size=1000,stat_cache_expire=900,retries=5,connect_timeout=10

# FROM ubuntu:14.04
# WORKDIR /dataset
# COPY --from=0 /data ./
# CMD /bin/bash 

