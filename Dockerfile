FROM python:3

ADD . /

RUN pip install dependency_injector
RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install mlxtend
RUN pip install mnist

ENTRYPOINT [ "python", "./main.py" ]