FROM allennlp/allennlp:v0.8.3

# install source code
ADD ./ /root/SPM/
WORKDIR /root/SPM/
RUN pip install -r requirements.txt
