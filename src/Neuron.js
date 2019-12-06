const Activations=require("./Activation.js");

function check_obj(obj){
	if(obj==undefined){
		return false;
	}else{
		if(obj.acc!=null){ return true;}else{return false;}
	}
}

class Connection{

	constructor(connectedNeuron,acc,util)
	{
		this.connectedNeuron = connectedNeuron;
		if(acc==null) {
			this.weight = 0.0;
			this.dWeight = 0.0;
		}else{
			this.weight=acc.zero_arr;
			this.dWeight=acc.zero_arr;
		}
		this.acc=acc;
		this.util=util;
	}
}

/**
* Creates a Neuron
* @param {array} layer  - The amount of neurons for each layer.
* @param {array} activation  - The activation function for each layer.
* @param {json} param  - Accepts the JSON data type.
*@returns {Network} Neuron Object
*/
class Neuron{

	constructor(layer,activation,param,acc,util){
        //optimize in next update..
		this.dendrons = [];
		if(acc==null) {
			this.eta = 0.001;
			this.alpha = 0.01;
			this.error = 0.0;
			this.gradient = 0.0;
			this.output = 0.0;
		}else{
			this.eta =      acc.zero_zero_zero_one;
			this.alpha =    acc.zero_zero_one;
			this.error =    acc.zero_arr;
			this.gradient = acc.zero_arr;
			this.output =   acc.zero_arr;
		}
		this.acc=acc;
		this.util=util;
		this.activation=activation;
		this.param=param;
		if (layer != null) {
			layer.forEach(function(neuron) {
				var con=new Connection(neuron,neuron.acc,neuron.util);
				this.dendrons.push(con);
			},this);
		}
	}
    /**
    * Add error in a specific neuron
    * @param {number} err 
    */
	addError(err)
	{

		if(check_obj(this)){
			this.error = this.util.add(this.error,err);
		}else{
			this.error = this.error+err;
		}
	}

    /**
    * Set error in a specific neuron
    * @param {number} err 
    */
	setError(err)
	{
		this.error = err;
	}

    /**
    * Add error in a specific neuron
    * @param {number} err 
    */
    
	setOutput(output)
	{
		this.output = output;
	}

    /**
    * Get the array expected by the neuron.
    * @returns {array} output  
    */
    
	getOutput()
	{
		return this.output;
	}

    /**
    * Perform Feed forward operation in Neural Network.
    */
    
	feedForward()
	{

		var sumoutput;
		if(!check_obj(this)) {
			sumoutput = 0;
		}else {
			sumoutput = this.acc.zero_arr;
		}
		if (this.dendrons.length == 0) {
			return;
		}
		var use_gpu=check_obj(this);
		this.dendrons.forEach(function(dendron) {
			if(use_gpu){
				var val=dendron.util.linear_mul(dendron.connectedNeuron.getOutput(),dendron.weight);
				sumoutput = dendron.util.add(sumoutput,val);
			}else{
				var val=dendron.connectedNeuron.getOutput()*dendron.weight;
				sumoutput = sumoutput+val;
			}
		});
		var activation=new Activations(this.acc,this.util);
		if(this.activation=="sigmoid"){
			this.output = activation.sigmoid(sumoutput);
		}else if(this.activation=="relu"){
			this.output = activation.relu(sumoutput);
		}else{
			this.output = activation.leaky_relu(sumoutput);
		}
	}

    /**
    * Assign weights to the neuron
    * @param {array}  
    */
	assign_weights(arr){
		var ctr=0;
		if(!check_obj(this)) {
			this.dendrons.forEach(function (dendron) {
				dendron.weight = arr[ctr];
				ctr++;
			});
		}else{
			this.dendrons.forEach(function (dendron) {
				dendron.weight = this.acc.define_array(arr[ctr]);
				ctr++;
			});
		}
	}

    /**
    * Get weights to the neuron
    * @returns {array}  
    */
	get_weights(){
		var weight=[];
		if(!check_obj(this)) {
			this.dendrons.forEach(function (dendron) {
				weight.push(dendron.weight);
			});
		}else{
			this.dendrons.forEach(function (dendron) {
				weight.push(dendron.acc.get_array(dendron.weight));
			});
		}
		return weight;
	}

    /**
    * To perform backpropogation algorithm.
    */
    
	backPropogate()
	{
		var activation=new Activations(this.acc,this.util);
		if(this.activation=="sigmoid"){
			if(check_obj(this)==true){
				this.gradient = this.util.linear_mul(this.error,activation.dsigmoid(this.output));
			}else{
				this.gradient = this.error*activation.dsigmoid(this.output);
			}
		}else if(this.activation=="relu"){
			if(check_obj(this)==true){
				this.gradient = this.util.linear_mul(this.error,activation.drelu(this.output));
			}else{
				this.gradient = this.error*activation.drelu(this.output);
			}
		}else{
			if(check_obj(this)==true){
				this.gradient = this.util.linear_mul(this.error,activation.d_leaky_relu(this.output));
			}else{
				this.gradient = this.error*activation.d_leaky_relu(this.output);
			}
		}
		this.dendrons.forEach(function(dendron) {
			if(check_obj(this)==true){
				dendron.dWeight = this.util.linear_mul(dendron.connectedNeuron.output,this.gradient);
				dendron.dWeight = this.util.linear_mul(this.eta,dendron.dWeight);
				var num=this.util.linear_mul(this.alpha,dendron.dWeight);
				dendron.dWeight=this.util.add(dendron.dWeight,num);
				dendron.weight = this.util.add(dendron.weight,dendron.dWeight);
			}else{
				dendron.dWeight = dendron.connectedNeuron.output*this.gradient;
				dendron.dWeight = this.eta*dendron.dWeight;
				var num=this.alpha*dendron.dWeight;
				dendron.dWeight=dendron.dWeight+num;
				dendron.weight = dendron.weight+dendron.dWeight;
			}
			if(this.param!=null){
				if(this.param["learning_rate"]!==undefined){
					if(check_obj(this)){
						var result=dendron.util.sub(dendron.weight,dendron.util.linear_mul(this.param["learning_rate"],this.gradient));
						dendron.connectedNeuron.addError(result);
					}else {
						dendron.connectedNeuron.addError((dendron.weight - this.param["learning_rate"]) * this.gradient);
					}
				}else{
					if(check_obj(this)) {
						dendron.connectedNeuron.addError(dendron.util.linear_mul(dendron.weight,this.gradient));
					}else {
						dendron.connectedNeuron.addError(dendron.weight * this.gradient);
					}
				}
			}else{
				if(check_obj(this)) {
					dendron.connectedNeuron.addError(dendron.util.linear_mul(dendron.weight,this.gradient));
				}else {
					dendron.connectedNeuron.addError(dendron.weight * this.gradient);
				}
			}
		},this);
		if(check_obj(this)) {
			this.error = this.acc.zero_arr;
		}else{
			this.error = 0;
		}
	}


}
module.exports = Neuron
