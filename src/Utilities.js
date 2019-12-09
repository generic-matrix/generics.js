const fs = require('fs');
const Activations=require("./Activation.js");
const Network=require("./ANN.js");

function arr_split(longArray,size){
	return new Array(Math.ceil(longArray.length / size)).fill("")
		.map(function() { return this.splice(0, size) }, longArray.slice());
}
function map_with_index(arr,index_arr){
	var res=[];
	index_arr.forEach(function(index){
		res.push(arr[index]);
	});
	return res;
}
Array.prototype.remove = function (v) {
	if (this.indexOf(v) != -1) {
		this.splice(this.indexOf(v), 1);
		return true;
	}
	return false;
}

function shuffleArray(array1,array2) {
	for (let i = array1.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[array1[i], array1[j]] = [array1[j], array1[i]];
		[array2[i], array2[j]] = [array2[j], array2[i]];
	}
}

/**
    *  A class which helps as a driver core to se the neural network with ease.
*/
class Utilities{

	constructor(Accelerator=null,settings=null) {
		if(Accelerator!=null){
			this.Accelerator=Accelerator;
			this.settings=settings;
			this.acc = new Accelerator.accelerator(settings);
			this.acc_util = new Accelerator.util(settings);
		}else{
			this.acc = null;
			this.acc_util = null;
		}
	}
	SIGMOID() { return "sigmoid"; }
	RELU() { return "relu"; }
	LEAKY_RELU() { return "leaky_relu"; }
    
    /**
    * Saves the model in a directory. 
    * @param {Network}  net
    * @param {string}  Directory: to save the ANN in a directory.
    */
	save_model(net,dir){
		var json={};
		json.topology=net.topology;
		json.activations=net.activations;
		if(this.acc!=null){
			var param=net.param;
			if(typeof(param.learning_rate)!="number") {
				param.learning_rate = this.acc.get_array(param.learning_rate)[0];
			}
			json.param = param;
		}else {
			json.param = net.param;
		}
		var weight = [];
		net.layers.forEach(function(layer){
			var lay=[];
			layer.forEach(function(neuron){
				lay.push(neuron.get_weights());
			});
			weight.push(lay);
		});
		json.weights=weight;

		try{
			fs.writeFileSync(dir,JSON.stringify(json), 'utf8');
			console.log("Saved the model "+dir);
		}catch(err){
			throw new Error(err);
		}
	}

	convert_to_tensors(arr){
		var array=[];
		for(var i=0;i<arr.length;i++) {
			array.push(this.acc.define_array(arr[i]));
		}
		return array;
	}
    /**
    * To train a neural network - Takes the entire batch size
    * @param {Network}  net
    * @param {string}  inputs: The x axis of the data set.
    * @param {string}  output: The y axis of the data set.
    * @param {string}  count:  The count value.
    */
	train(net,inputs,output,count){
		if(this.acc!=null) {
			//recusively convert into tensors..
			var input_arr=[];
			var output_arr=[];
			for(var i=0;i<inputs.length;i++){
				input_arr.push(this.convert_to_tensors(inputs[i]));
				output_arr.push(this.convert_to_tensors(output[i]));
			}
			for (var j = 0; j < count; j++) {
				var erro = 0;
				for (var i = 0; i < input_arr.length; i++) {
					net.setInput(input_arr[i]);
					net.feedForward();
					net.backPropogate(output_arr[i]);
					erro = erro + net.getError(output_arr[i]);
				}
				try {
					process.stdout.cursorTo(0);
					process.stdout.write("Error: " + erro);
				}catch(err) {
					console.log("Error: " + erro);
				}
			}
		}else {
			for (var j = 0; j < count; j++) {
				var erro = 0.0;
				for (var i = 0; i < inputs.length; i++) {
					net.setInput(inputs[i]);
					net.feedForward();
					net.backPropogate(output[i]);
					erro = erro + net.getError(output[i]);
				}
				try {
					process.stdout.cursorTo(0);
					process.stdout.write("Error: " + erro);
				} catch (err) {
					console.log("Error: " + erro);
				}
			}
		}

	}

    /**
    * To get predictions from the neural network.
    * @param {Network}  net
    * @param {string}  inputs: The x axis of the data set.
    * @param {input_array}  input_array: The x axis of the data set.
    * @returns {array}  result:  The result predicted by the network.
    */
	predict(net,input){
		if(this.acc!=null) {
			net.setInput(this.convert_to_tensors(input));
			net.feedForward();
			return net.getTheResults();
		}else {
			net.setInput(input);
			net.feedForward();
			return net.getTheResults();
		}
	}

    /**
    * To get restore model from the JSON file.
    * @param {string}  dir: To the JSON file.
    */
    
	restore_model(dir){
        //,use_gpu --> int he next update.
        var obj=this;
		return new Promise(function(resolve, reject) {
			fs.readFile(dir, function read(err, data) {
				if (err) {
					reject(err);
				}
				var json=JSON.parse(data);
				var topology=json["topology"];
				var act=json["activations"];
				var activations=[];
				var util=new Utilities();
				act.forEach(function(layer){
					if(layer=="sigmoid"){
						activations.push(util.SIGMOID());
					}else if(layer=="relu"){
						activations.push(util.RELU());
					}else if(layer=="leaky_relu"){
						activations.push(util.LEAKY_RELU());
					}
				});
				var weights=json["weights"];
				var param=json["param"];
				if(obj.acc!=null){
					var net = new Network(topology, activations, param,obj.Accelerator,obj.settings);
				}else {
					var net = new Network(topology, activations, param);
				}
				net.assign_weights(weights);
				resolve(net);
			});
		});
	}
    
     /**
    * To test the neural network predictions with the expected predictions.
    * @param {Network}  net : The model object.
    * @param {array}  x_test_axis : The x_axis as the input for which we need the predictions.
    * @param {array}  y_test_axis : The y_axis is the output the model is intended to do.
    * @param {JSON}  json : The JSON must contain the keys "g_x_plots" , "e_x_plots" and "y_plots" all array paramas.
    * @param {number}  step : A numerical value 
    * @param {number}  threashold : The threashold value ,if the predicted value is less  than it then success else it's counted as failure.
    */
    
	test(net,x_test_axis,y_test_axis,json,step,threashold){
		var ctr=0;
		var success=0;
		var failure=0;
		var obj=this;
		if(obj.acc!=null){
			var acc_t=obj.acc.define_array(threashold);
		}
		x_test_axis.forEach(function(input_arr){
			if(obj.acc!=null){
				net.setInput(obj.acc.define_array(input_arr));
			}else{
				net.setInput(input_arr);
			}
			net.feedForward();
			if(obj.acc!=null) {
				var val1 = obj.acc_util.abs(obj.acc.define_array(net.getTheResults()));
				var val2 = obj.acc_util.abs(obj.acc.define_array(y_test_axis[ctr]));
				json.g_x_plots.push(val1);
				json.e_x_plots.push(val2);
				json.y_plots.push(step + ctr);
				var dif = obj.acc_util.abs(obj.acc_util.sub(val1,val2));
				if (obj.acc_util.linear_max_boolean( obj.acc_util.abs(dif),acc_t)) {
					failure++;
				} else {
					success++;
				}
				console.log("\n Given: "+obj.acc.get_array(val1).toString()+" \n Expected: "+obj.acc.get_array(val2).toString()+"\n");
			}else {
				var val1 = Math.abs(net.getTheResults()[0]);
				var val2 = Math.abs(y_test_axis[ctr][0]);
				json.g_x_plots.push(val1);
				json.e_x_plots.push(val2);
				json.y_plots.push(step + ctr);
				var dif = Math.abs(val2 - val1);
				if (Math.abs(dif) > threashold) {
					failure++;
				} else {
					success++;
				}
				console.log("Given: "+val1+" Expected: "+(val2)+" Difference: "+dif);
			}
			ctr++;
		});

		console.log("Correct : "+success+" -- Failed : "+failure);

	}

     /**
    * To test the neural network predictions with the expected predictions.
    * @param {Network}  net : The model object.
    * @param {array}  input : The x_axis as the input for which we need the predictions.
    * @param {array}  output : The y_axis is the output the model is intended to do.
    * @param {number}  folds : By what size the data set must be folded ,defins the batch size.
    * @param {number}  count  : How many times a neural network must be trained .
    * @param {string}  dir : The  directory where the model must be saved.
    * @param {number}  threashold : The threashold value ,if the predicted value is less  than it then success else it's counted as failue.
    * @param {number}  percent: The percentage bywhich a input data must be divided in a batch size.
    */
    
	perform_k_fold(net,input,output,folds,count,dir,threashold,percent){
		var obj=this;
		return new Promise(function(resolve, reject) {
			if(input.length!=output.length){
				reject("The x axis and the y axis length is not same");
			}
			if(percent<=0||percent>=100){
				reject("The split percent is invalid.");
			}

			var json={};
			json.title="Metrics";
			json.x_axis="Steps";
			json.y_axis="predictions";
			json.e_x_plots=[];
			json.g_x_plots=[];
			json.y_plots=[];

			if(folds==0||folds.length>=input.length){
				reject("Fold value is invalid.");
			}else{
				shuffleArray(input,output);
				var x_data_set=arr_split(input,folds);
				var y_data_set=arr_split(output,folds);
				for(let j=0;j<x_data_set.length;j++){
					var index=Math.round((percent/100)*x_data_set[j].length);
					var x_train=x_data_set[j];
					var y_train=y_data_set[j];
					var x_test=x_train.splice(0,index);
					var y_test=y_train.splice(0,index);
					obj.train(net,x_train,y_train,count);
					obj.test(net,x_test,y_test,json,j,threashold);
					obj.save_model(net,dir);
				}

			}
			resolve(json);
		});
	}

}
module.exports = Utilities
