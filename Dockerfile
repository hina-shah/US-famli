
FROM tensorflow/tensorflow:1.15.2-py3-jupyter

WORKDIR /usr/src/US-famli
COPY . .

RUN pip install --no-cache-dir -r requirements.txt && \
	apt-get update && apt-get install -y tesseract-ocr && apt-get clean && \
	apt-get install -y git && \
	apt-get install -y git-lfs && git lfs install && \
	mkdir /usr/.nvm && \
	export NVM_DIR="/usr/.nvm" && \
	curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash && \
	[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && \
	[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion" && \
	nvm install --lts && \
	export NODE_VER=`node --version` && \
	ln -s /usr/.nvm/versions/node/$NODE_VER/bin/node /usr/bin/ && \
	ln -s /usr/.nvm/versions/node/$NODE_VER/bin/npm /usr/bin/ && \
	chmod ugo+x /usr/bin/node && \
	chmod ugo+x /usr/bin/npm && \
	mkdir /.npm && chmod -R a+rw /.npm

	## Uncomment this to add the node code and move to the end of the last line
	# && \
	# cd .. && git clone https://github.com/juanprietob/us-famli-nn.github && \
	# cd us-famli-nn/models && tar -xf

# Uncomment this to add the node code
# WORKDIR /usr/src/us-famli-nn
# && npm install
