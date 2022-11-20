# FROM ubuntu:latest
FROM python:3.8.1
COPY "run.sh" .
RUN ["chmod", "+x", "./run.sh"]
COPY mlops /exp/mlops
COPY ./requirements.txt /exp/requirements.txt
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
ENTRYPOINT [ "./run.sh" ]
#CMD ["python3", "./mlops/plot_graphs.py"]