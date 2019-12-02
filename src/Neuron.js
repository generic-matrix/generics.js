const Activations=require("./Activation.js");

function check_obj(obj){
	if(obj==undefined){
		return false;
	}else{
		if(obj.custom_adapter!=null){ return true;}else{return false;}
	}
}
class Connection{

	constructor(connectedNeuron,param,optimize)
	{
		this.connectedNeuron = connectedNeuron;
		this.weight = 0.0;
		this.dWeight = 0.0;
		this.optimize=optimize;
		this.param=param;
	}
}

class Neuron{
	
	constructor(layer,activation,param,optimize){
		this.dendrons = [];
		this.eta = 0.001;
		this.alpha = 0.01;
		this.error = 0.0;
		this.gradient = 0.0;
		this.output = 0.0;
		this.optimize=optimize;
		this.activation=activation;
		this.param=param;
		if (layer != null) {
			layer.forEach(function(neuron) {
				var con=new Connection(neuron,optimize);
				this.dendrons.push(con);
			},this);
		}
	}
	
	addError(err)
	{
		if(check_obj(this)==true){
			this.error = this.custom_adapter.add(this.error,err);
		}else{
			this.error = this.error+err;
		}
	}

	
	setError(err)
	{
		this.error = err;
	}

	setOutput(output)
	{
		this.output = output;
	}

	getOutput()
	{
		return this.output;
	}
	
	feedForward()
	{
		var sumoutput = 0;
		if (this.dendrons.length == 0) {
			return;
		}
		
		this.dendrons.forEach(function(dendron) {
			if(check_obj(this)==true){
			var val=this.custom_adapter.multiply(dendron.connectedNeuron.getOutput(),dendron.weight);
			sumoutput =  Adapter.add(sumoutput,val);
			}else{
			var val=dendron.connectedNeuron.getOutput()*dendron.weight;
			sumoutput = sumoutput+val;	
			}
		});
		var activation=new Activations(this.optimize);
		if(this.activation=="sigmoid"){
			this.output = activation.sigmoid(sumoutput);
		}else if(this.activation=="relu"){
			this.output = activation.relu(sumoutput);
		}else{
			this.output = activation.leaky_relu(sumoutput);
		}
	}
	
	assign_weights(arr){
		var ctr=0;
		this.dendrons.forEach(function(dendron) { dendron.weight=arr[ctr]; ctr++; });
	}
	
	get_weights(){
		var weight=[];
		var sumoutput = 0;
		this.dendrons.forEach(function(dendron) { weight.push(dendron.weight); });
		return weight;
	}
	
	backPropogate()
	{
		var activation=new Activations(this.optimize);
		if(this.activation=="sigmoid"){
		if(check_obj(this)==true){
			this.gradient = this.custom_adapter.multiply(this.error,activation.dsigmoid(this.output));
		}else{
			this.gradient = this.error*activation.dsigmoid(this.output);
		}
		}else if(this.activation=="relu"){
		if(check_obj(this)==true){
			this.gradient = this.custom_adapter.multiply(this.error,activation.drelu(this.output)); 
		}else{
			this.gradient = this.error*activation.drelu(this.output); 
		}  
		}else{
		if(check_obj(this)==true){
			this.gradient = this.custom_adapter.multiply(this.error,activation.d_leaky_relu(this.output)); 
		}else{
			this.gradient = this.error*activation.d_leaky_relu(this.output); 
		}   
		}
		this.dendrons.forEach(function(dendron) { 
			if(check_obj(this)==true){
				dendron.dWeight = (this.custom_adapter.multiply(dendron.connectedNeuron.output,this.gradient));
				dendron.dWeight = this.custom_adapter.multiply(this.eta,dendron.dWeight);
				var num=this.custom_adapter.multiply(this.alpha,dendron.dWeight);
				dendron.dWeight=this.custom_adapter.add(dendron.dWeight,num);
				dendron.weight = this.custom_adapter.add(dendron.weight,dendron.dWeight);
			}else{
				dendron.dWeight = dendron.connectedNeuron.output*this.gradient;
				dendron.dWeight = this.eta*dendron.dWeight;
				var num=this.alpha*dendron.dWeight;
				dendron.dWeight=dendron.dWeight+num;
				dendron.weight = dendron.weight+dendron.dWeight;
			}
			if(this.param!=null){
				if(this.param["learning_rate"]!=undefined){
					dendron.connectedNeuron.addError((dendron.weight - this.param["learning_rate"] ) *  this.gradient);
				}else{
					dendron.connectedNeuron.addError(dendron.weight *  this.gradient);
				}
			}else{
				dendron.connectedNeuron.addError(dendron.weight *  this.gradient);
			}
		},this);
		this.error = 0;
	}

	
}
module.exports = Neuron