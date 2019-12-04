# A minimal deep learning library for the web
generics.js (0.1.0)

![alt text](https://www.trygistify.com/generics.js/assets/icons/logo.png)

The library allows to leverage to create and deploy real time deep learning solution currently including ANN and CNN with fully featured reinforcement learning and k-fold cross validation tests.

## API Docs :
www.trygistify.com/generics

## Real time examples:
Food rating prediction: 

Dogs and cats prediction: 

## Pull it using npm:
`npm install generics.js --save`

## Manual installation:
```
git clone https://github.com/generic-matrix/generics.js.git
unzip generics.js.zip
cd generics.js && npm install -g --save

```

## Use it as:
```
let gen = require("generics.js");
```
## Features  :
1) ### K fold cross validation tests  
(used to evaluate machine learning models on a limited data sample) :  
```
var dir = "/content/my_model.json";
var summary_url = "/content/summary.json";
var util = new gen.Utilities();
var training_count = 10;
var batch_size = 10;
var testing_threashold = 0.45;
var split_percent = 20;
var util = new gen.Utilities();
var topology=[];
topology.push(x_axis[0].length);
topology.push(50);
topology.push(y_axis[0].length);
var activations = [];
activations.push(util.LEAKY_RELU());
activations.push(util.LEAKY_RELU());
activations.push(util.SIGMOID());
var param={
    "learning_rate":0.1
};

var net=new gen.Network(topology,activations,param,null);
util.perform_k_fold(net, x_axis, y_axis, batch_size, training_count, dir, testing_threashold, split_percent);
```
  
2) ### Easy retriving of model  :  

```
var net = null;
var util = new gen.Utilities();
var dir2 = "/content/my_model.json";
util.restore_model(dir2).then(function(net2){
     net=net2;
});

```
3) ### Inbuild CSV parsing :
Refer: https://www.trygistify.com/generics#preprocessingparse_csv
```
var pre=new gen.Pre_Processing();
var fill_type = 0;
pre.parse_csv("/content/cereal.csv", fill_type, ["mfr", "type", "calories", "protein", "fat", "sodium", "fiber", "carbo", "sugars", "potass", "vitamins", "shelf", "weight", "cups"], ["rating"])
.then(function (json) {
  console.log(json);
});
```

## License :

https://github.com/generic-matrix/generics.js/blob/master/LICENSE

Logo icon for generics.js made by [Good Ware](https://www.flaticon.com/authors/good-ware) from www.flaticon.com

