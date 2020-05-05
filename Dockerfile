
FROM tensorflow/tensorflow:1.15.2-gpu-py3-jupyter

WORKDIR /usr/src/US-famli
COPY . .

RUN pip install --no-cache-dir -r requirements.txt && \
	apt-get update && apt-get install -y tesseract-ocr && apt-get clean && \
	apt-get install -y git && \
	apt-get install -y git-lfs && git lfs install

	## Uncomment this to add the node code and move to the end of the last line
	# && \
	# cd .. && git clone https://github.com/juanprietob/us-famli-nn.github && \
	# cd us-famli-nn/models && tar -xf

# Uncomment this to add the node code
# WORKDIR /usr/src/us-famli-nn

SHELL ["/bin/bash", "--login", "-i", "-c"]
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash
RUN source /root/.bashrc && nvm install --lts

# Uncomment this to add the node code and append to the last line
# && npm install

SHELL ["/bin/bash", "--login", "-c"]