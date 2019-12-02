

function check_obj(obj){
	if(obj==undefined){
		return false;
	}else{
		if(obj.custom_adapter!=null){ return true;}else{return false;}
	}
}

class Activations{
	constructor(optimize){
		this.optimize=optimize;
	}
	sigmoid(x)
		{
			if(check_obj(this)){
			var number= this.custom_adapter.exp(-x*1.000);
			var num=this.custom_adapter.add(1,number);
			return 1 / num;
			}else{
			return 1 / (1 + Math.exp(-x * 1.0));
			}
		}

		dsigmoid(x)
		{
			if(check_obj(this)){
			var num=this.custom_adapter.multiply(x,(1.000 - x));
			return num;
			}else{
			return x*(1.00-x);
			}
		}
		
		relu(x)
		{
			return Math.max(0,x);
		}

		drelu(x)
		{
			if(x>0){return x;}else{return (0.1*x);}
		}
		
		leaky_relu(x)
		{
			if(x<0){
				var num=-1;
				if(check_obj(this)){
				num=this.custom_adapter.multiply(0.01,x);
				}else{
				num=0.01*x;	
				}
				return num;
			}else{
				return x; 
			}
		}

		d_leaky_relu(x)
		{
			if(x<0){
				return 0.01;
			}else{
				return 1; 
			}
		}
}
module.exports = Activations