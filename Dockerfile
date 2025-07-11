#FROM python:3.13-bookworm
FROM minizinc/minizinc:2.9.3-noble

RUN apt update && \
    apt install -y python3-pip python3-venv coinor-cbc glpk-utils && \
    rm -rf /var/lib/apt/lists/*
# Since python installed by apt is externally managed we must create a virtual environment
RUN python3 -m venv /venv
WORKDIR /app
COPY requirements.txt /app/
RUN /venv/bin/pip install -r /app/requirements.txt
COPY Instances /app/Instances
COPY run.py /app/
COPY CP /app/CP
COPY SMT /app/SMT
COPY MIP /app/MIP

CMD [ "/venv/bin/python", "/app/run.py" ]
