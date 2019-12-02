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

class Utilities{
	
	SIGMOID() { return "sigmoid"; }
	RELU() { return "relu"; }
	LEAKY_RELU() { return "leaky_relu"; }
	save_model(net,dir){
		var json={};
		json.topology=net.topology;
		json.activations=net.activations;
		json.param=net.param;
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
		
	train(net,inputs,output,count){
			for(var j=0;j<count;j++){
					var erro = 0.0;
					for (var i = 0; i < inputs.length ; i++) {
							net.setInput(inputs[i]);
							net.feedForward();
							net.backPropogate(output[i]);
							erro = erro + net.getError(output[i]);							
					}
					try{
					process.stdout.cursorTo(0);
					process.stdout.write("Error: "+erro);
					}catch(err){
						console.log("Error: "+erro);
					}
			}
			
		}
	
	predict(net,input_arr){
		net.setInput(input_arr);
		net.feedForward();
		return net.getTheResults();
	}	
	
	restore_model(dir,use_gpu){
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
				var net=new Network(topology,activations,param,use_gpu);
				net.assign_weights(weights);
				resolve(net);
			});
		});
	}
	test(net,x_test_axis,y_test_axis,json,step,threashold){
			var ctr=0;
			var success=0;
			var failure=0;
			x_test_axis.forEach(function(input_arr){
			net.setInput(input_arr);
			net.feedForward();
			var val1=Math.abs(net.getTheResults()[0]);
			var val2=Math.abs(y_test_axis[ctr][0]);
			json.g_x_plots.push(val1);
			json.e_x_plots.push(val2);
			json.y_plots.push(step+ctr);
			var dif=Math.abs(val2-val1);
			if(Math.abs(dif)>threashold){
				failure++;
			}else{
				success++;
			}
			console.log("Given: "+val1+" Expected: "+(val2)+" Difference: "+dif);
			ctr++;
		});
		
		console.log("Correct : "+success+" -- Failed : "+failure);
			
	}
	
	perform_k_fold(net,input,output,folds,count,dir,threashold,percent){

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
					//shuffleArray(input,output);
					var x_data_set=arr_split(input,folds);
					var y_data_set=arr_split(output,folds);
					for(let j=0;j<x_data_set.length;j++){
						var index=Math.round((percent/100)*x_data_set[j].length);
						var x_train=x_data_set[j];
						var y_train=y_data_set[j];
						var x_test=x_train.splice(0,index);
						var y_test=y_train.splice(0,index);
						var util=new Utilities();
						util.train(net,x_train,y_train,count);
						util.test(net,x_test,y_test,json,j,threashold);
						util.save_model(net,dir);
					}
				
				}
				resolve(json);
		  });
	}
		
}
module.exports = Utilities