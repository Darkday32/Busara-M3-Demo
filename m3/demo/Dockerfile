# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

RUN apt-get update
RUN apt-get install -y wget python3.10 python3.10-dev git unzip 
RUN apt-get install -y python3-pip

SHELL ["/bin/bash", "-c"] 
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN git clone https://github.com/Project-MONAI/VLM --recursive
WORKDIR /VLM
RUN make demo_m3
EXPOSE 8080
RUN cd m3/demo
CMD ["python", "gradio_m3.py"]
