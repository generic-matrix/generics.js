const Neuron=require("./Neuron.js");

/**
* Creates a Neural Network based on the paramas
* @param {array} topology  - The amount of neurons for each layer.
* @param {array} activations  - The activation function for each layer.
* @param {array} param  - Accepts the JSON data type.
*@returns {Network} Network Object
*/


class Network{
	//this.custom_adapter will be in the next update.
 	constructor(topology,activations,param,Accelerator=null,settings=null){
		this.layers = [];
		this.topology=topology;
		this.activations=activations;
		this.param=param;
		if(Accelerator!=null) {
			this.acc = new Accelerator.accelerator(settings);
			this.acc_util = new Accelerator.util(settings);
			if(this.param["learning_rate"]!=undefined) {
				this.param["learning_rate"] = this.acc.define_array([this.param["learning_rate"]]);
			}
		}else{
			console.error("-------------------------------------------------------------------------------------------------");
			console.error("| You can install accelerator.js by [ npm install accelerator.js -g --save ] to leverage the GPU|");
			console.error("-------------------------------------------------------------------------------------------------");
			this.acc = null;
			this.acc_util = null;
		}
		topology.forEach(function(numNeuron) {
			var layer=[];
			for(var i=0;i<numNeuron;i++){
				if (this.layers.length == 0) {
					layer.push(new Neuron(null,activations[i],param,this.acc,this.acc_util));
				} else {
					layer.push(new Neuron(this.layers[this.layers.length - 1],activations[i],param,this.acc,this.acc_util));
				}
			}
			if(this.acc!=null){
				let one_arr=this.acc.one_arr;
				layer[layer.length - 1].setOutput(one_arr);
			}else {
				layer[layer.length - 1].setOutput(1);
			}
			this.layers.push(layer);
		},this);
	}
	setInput(inputs)
	{
		for (var i = 0;i< inputs.length;i++) {
			if(this.acc!=null) {
				this.layers[0][i].setOutput(this.acc.define_array([inputs[i]]));
			}else{
				this.layers[0][i].setOutput(inputs[i]);
			}
		}
	}

	setOutput(output)
	{
		this.output = output;

	}

    /*
    Feeds forward in the neural network.
    */
	feedForward()
	{
		var layers = this.layers.slice(1);
		layers.forEach(function(layer) {
			layer.forEach(function(neuron) {
				neuron.feedForward();
			});
		});
	}

    /**
    * Performs backpropogation algorithm in the neural network
    * @param {array} target  - Array
    */
	backPropogate(target)
	{
		for (var i = 0; i < target.length; i++) {
			if(this.acc!=null){
				var temp=this.layers[this.layers.length - 1][i].getOutput();
				var res = this.acc_util.sub(this.acc.define_array([target[i]]),temp);
				this.layers[this.layers.length - 1][i].setError(res);
			}else{
				var res = target[i]-this.layers[this.layers.length - 1][i].getOutput();
				this.layers[this.layers.length - 1][i].setError(res);
			}
		}
		this.layers.reverse().forEach(function(layer) {
			layer.forEach(function(neuron){
				neuron.backPropogate();
			});
		});
		this.layers.reverse()
	}

    /**
    * Gets the error in the neural network model.
    * @param {array} target  - Array
      @returns {number} The error value
    */
    
	getError(target)
	{
		var err = 0;
		for (var i = 0; i < target.length; i++) {
			if(this.acc!=null){
				var e=this.acc_util.sub(this.acc.define_array([target[i]]),this.layers[this.layers.length - 1][i].getOutput());
				var pow=this.acc_util.pow(e,this.acc.two_arr);
				err=this.acc.get_array(pow)[0]+err;
			}
			else{
				let e = target[i]-this.layers[this.layers.length - 1][i].getOutput();
				err = err + Math.pow(e,2);
			}
		}
		err =  err / target.length;
		err = Math.sqrt(err);
		return err;
	}
     /**
    * Assign weight based on the given JSON ,used to retain the model.
    * @param {json} json  
    */
	assign_weights(json){
		var i=0;
		this.layers.forEach(function(layer){
			var j=0;
			layer.forEach(function(neuron){
				neuron.assign_weights(json[i][j]);
				j++;
			});
			i++;
		});
	}


 /**
* Get the results from the neural network based on the output dimentions.
* @returns {array} output based on the model output dimentions.
*/
getTheResults()
	{
		var output = [];
		if(this.acc!=null){
			this.layers[this.layers.length - 1].forEach(function (neuron) {
				var o = neuron.acc.get_array(neuron.getOutput());
				output.push(o);
			});
			return output;
		}else {
			this.layers[this.layers.length - 1].forEach(function (neuron) {
				var o = neuron.getOutput();
				output.push(o);
			});
			return output;
		}
	}
}
module.exports = Network
