from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)
model_path = "svm_gamma=0.0005_C=2.joblib"
model = load(model_path)

@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Hello, World!</b>"


# get x and y somehow    
#     - query parameter
#     - get call / methods
#     - post call / methods ** 

@app.route("/sum", methods=['POST'])
def sum():
    x = request.json['x']
    y = request.json['y']
    z = x + y 
    return {'sum':z}



@app.route("/predict", methods=['POST'])
def predict_digit():
    image = request.json['image']
    print("done loading")
    predicted = model.predict([image])
    return {"y_predicted":int(predicted[0])}


@app.route("/compare", methods=['POST'])
def compare_images():
    
    image1 = request.json['image1']
    image2 = request.json['image2']
    
    print("done loading images")
    first_img = model.predict([image1])
    second_img = model.predict([image2])
    if first_img == second_img:
       return {"result":"Images are same"}
    else:
       return {"result":"Images are not same"}
