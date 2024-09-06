FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

# Install SSH server
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd

# Set root password
RUN echo 'root:abcd' | chpasswd

# Allow root login with password
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Clone the specific branch (feat/docker) and install the Python package
RUN git clone -b feat/docker https://github.com/devcow85/lita.git /app/lita \
    && cd /app/lita \
    && pip install .

# Expose SSH port
EXPOSE 22

# Start SSH service
CMD ["/usr/sbin/sshd", "-D"]

