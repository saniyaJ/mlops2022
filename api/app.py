from flask import Flask
from flask import request
from joblib import load
import os 
app = Flask(__name__)

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
    #model = load("C:\\Users\\sj36580\\Downloads\\IIT Jodhpur\\Sem 2\\mlops\\major\\mlops2022\\models\\svm_gamma=0.001_C=0.2.joblib")
    
    model = load(request.json['model_name'])
    print('entered')
    if(model == ""):
        files = os.listdir(os.getcwd()+'\\results')
        fscore =0
        print(files)
        for fname in files:
            
            with open(os.getcwd()+'\\results\\'+fname,'r') as f:
                content = f.read()
                if (fscore < content[content.find('macro_f1')+11:10]):
                    fscore = content[content.find('macro_f1')+11:10]
                    m_path =content.find('model path')
                    eod =content.find('\'}')
                    model = load(content[m_path+13:eod])
    
    
    image = request.json['image']
    predicted = model.predict([image])
    print('Result of predicted image')
    return {"y_predicted":int(predicted[0])}


@app.route("/compare", methods=['POST'])
def compare_images():
    model = load(request.json['model_name'])
    image1 = request.json['image1']
    image2 = request.json['image2']
    
    print("done loading images")
    first_img = model.predict([image1])
    second_img = model.predict([image2])
    if first_img == second_img:
       return {"result":"Images are same"}
    else:
       return {"result":"Images are not same"}

#curl http://127.0.0.1:5000/predict -X POST  -H 'Content-Type: application/json' -d '{"model_name":"models/svm_gamma=0.001_C=0.2.joblib","image": ["0.0","0.0","0.0","11.999999999999982","13.000000000000004","5.000000000000021","8.881784197001265e-15","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999988","9.000000000000005","1.598721155460224e-14","0.0", "0.0","0.0","2.9999999999999925","14.999999999999979","15.999999999999998","6.000000000000022","1.0658141036401509e-14", "0.0","6.217248937900871e-15","6.999999999999987","14.99999999999998","15.999999999999996","16.0","2.0000000000000284", "3.552713678800507e-15","0.0","5.5220263365470826e-30","6.21724893790087e-15","1.0000000000000113","15.99999999999998", "16.0","3.000000000000022","5.32907051820075e-15","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0", "6.000000000000015","1.0658141036401498e-14","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0", "6.000000000000018","1.0658141036401503e-14","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999993", "10.00000000000001","1.7763568394002505e-14","0.0"]}'
#curl http://127.0.0.1:5000/predict -X POST  -H 'Content-Type: application/json' -d '{"model_name":"","image": ["0.0","0.0","0.0","11.999999999999982","13.000000000000004","5.000000000000021","8.881784197001265e-15","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999988","9.000000000000005","1.598721155460224e-14","0.0", "0.0","0.0","2.9999999999999925","14.999999999999979","15.999999999999998","6.000000000000022","1.0658141036401509e-14", "0.0","6.217248937900871e-15","6.999999999999987","14.99999999999998","15.999999999999996","16.0","2.0000000000000284", "3.552713678800507e-15","0.0","5.5220263365470826e-30","6.21724893790087e-15","1.0000000000000113","15.99999999999998", "16.0","3.000000000000022","5.32907051820075e-15","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0", "6.000000000000015","1.0658141036401498e-14","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0", "6.000000000000018","1.0658141036401503e-14","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999993", "10.00000000000001","1.7763568394002505e-14","0.0"]}'