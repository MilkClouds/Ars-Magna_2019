import cloud_keras as cloud
import sys, os
from PIL import Image
import numpy as np
# 명령줄에서 파일 이름 지정하기 --- (★1)
if len(sys.argv) <= 1:
    print("(<파일 이름>)")
    quit()
image_size = 50
categories = ['altocumulus','Altostratus','Cirrocumulus','Cirrostratus Nebulosus','cirrus fibratus','cumulonimbus','cumulus humilis','Nimbostratus','stratocumulus','Stratus Nebulosus']
name=['고적운','고층운','권적운','권층운','권운','적란운','적운','난층운','층적운','층운']

# 입력 이미지를 Numpy로 변환하기 --- (★2)
X = []
files = []
for fname in sys.argv[1:]:
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    in_data = np.asarray(img)
    X.append(in_data)
    files.append(fname)
X = np.array(X)
# CNN 모델 구축하기 --- (★3)
model = cloud.build_model(X.shape[1:])
model.load_weights("./image/cloud-model.hdf5")
# 데이터 예측하기 --- (★4)
html = ""
pre = model.predict(X)
for i, p in enumerate(pre):
    y = p.argmax()
    print("+입력:", files[i])
    print("|구름 이름:", categories[y])
    print("|구름 이름:", name[y])
    html += """
        <h3>입력:{0}</h3>
        <div>
          <p><img src="{1}" width=300></p>
          <p>구름 이름:{2}</p>
          <p>구름 이름:{3}</p>
        </div>
    """.format(os.path.basename(files[i]),
        files[i],
        categories[y],
        name[y])
# 리포트 저장하기 --- (★5)
html = "<html><body style='text-align:center;'>" + \
    "<style> p { margin:0; padding:0; } </style>" + \
    html + "</body></html>"
with open("cloud-result.html", "w") as f:
    f.write(html)