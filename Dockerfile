FROM allennlp/allennlp:v0.8.3
WORKDIR /root/SPM/
ADD ./requirements.txt /root/SPM
RUN pip install -r requirements.txt
ADD ./spm/ /root/SPM/spm