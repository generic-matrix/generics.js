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
 	constructor(topology,activations,param){
		this.layers = [];
		this.topology=topology;
		this.activations=activations;
		this.param=param;
		this.custom_adapter=null;
		topology.forEach(function(numNeuron) {
			var layer=[];
			for(var i=0;i<numNeuron;i++){
				if (this.layers.length == 0) {
					layer.push(new Neuron(null,activations[i],param,this.custom_adapter));
				} else {
					layer.push(new Neuron(this.layers[this.layers.length - 1],activations[i],param,this.custom_adapter));
				}
			}
			//layer.push(new Neuron(null,null));
			layer[layer.length - 1].setOutput(1);
			this.layers.push(layer);
		},this);
	}
	setInput(inputs)
	{
		for (var i = 0;i< inputs.length;i++) {
			this.layers[0][i].setOutput(inputs[i]);
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
			let res=-1;
			if(this.layers==true){
				res = Adapter.subtract([target[i]],[this.layers[this.layers.length - 1][i].getOutput()]);
			}else{
				res = target[i]-this.layers[this.layers.length - 1][i].getOutput();
			}
			this.layers[this.layers.length - 1][i].setError(res);
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
			if(this.custom_adapter!=null){
				var e=this.custom_adapter.subtract(target[i],this.layers[this.layers.length - 1][i].getOutput());
				err = err + this.custom_adapter.pow(e,2);
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
		this.layers[this.layers.length - 1].forEach(function(neuron){
			var o = neuron.getOutput();
			output.push(o);
		});
		return output;
	}
}
module.exports = Network
