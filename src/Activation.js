

function check_obj(obj){
	if(obj===undefined){
		return false;
	}else{
		if(obj.acc!=null){ return true;}else{return false;}
	}
}

/**
* A class to define the activation functions.
*/
class Activations{
	constructor(acc,util){
        //optimize in next update..
		this.acc=acc;
		this.util=util;
	}
    /**
    * Performs sigmoid 
    * @param {number}  value
    * @returns {number}  Result
    */
	sigmoid(x)
		{
			if(check_obj(this)){
				var denominator=this.util.add(this.acc.one_arr,this.util.exp(this.util.linear_mul(x,this.acc.minus_one_arr)));
				return this.util.linear_div(this.acc.one_arr,denominator);
			}else{
				return 1 / (1 + Math.exp(x * -1.0));
			}
		}
    
    /**
    * Performs derivaive of a sigmoid 
    * @param {number}  value
    * @returns {number}  Result
    */

		dsigmoid(x)
		{
			if(check_obj(this)){
			var num=this.util.linear_mul(x,this.util.sub(this.acc.one_arr,x));
			return num;
			}else{
			return x*(1.00-x);
			}
		}
    
     /**
    * Performs Relu activation function
    * @param {number}  value
    * @returns {number}  Result
    */
		
		relu(x)
		{
			if(check_obj(this)) {
				return this.util.linear_max(this.acc.zero_arr,x);
			}else{
				return Math.max(0, x);
			}
		}
    
    /**
    * Performs derivative of Relu activation function
    * @param {number}  value
    * @returns {number}  Result
    */

		drelu(x)
		{
			if(check_obj(this)) {
				if(this.util.linear_max_boolean(x,this.acc.zero_arr)===0){
					return x;
				}else{
					return this.util.linear_mul(this.acc.zero_one,x);
				}

			}else {
				if (x > 0) {
					return x;
				} else {
					return (0.1 * x);
				}
			}
		}
		
        /**
    * Performs Leaky Relu activation function
    * @param {number}  value
    * @returns {number}  Result
    */
		leaky_relu(x)
		{
			if(check_obj(this)) {
				let zero_arr=this.acc.zero_arr;
				if (this.util.linear_max_boolean(x,zero_arr)) {
					return this.util.linear_mul(this.acc.zero_zero_one,x);
				} else {
					return x;
				}
			}else{
				if (x < 0) {
					var num = 0.01 * x;
					return num;
				} else {
					return x;
				}
			}
		}

    /**
    * Performs derivative of Leaky Relu activation function
    * @param {number}  value
    * @returns {number}  Result
    */
		d_leaky_relu(x)
		{
			if(check_obj(this)) {
				let zero_arr=this.acc.zero_arr;
				if (this.util.linear_max_boolean(x,zero_arr)===1) {
					return this.acc.zero_zero_one;
				} else {
					return this.acc.one_arr;
				}
			}else {
				if (x < 0) {
					return 0.01;
				} else {
					return 1;
				}
			}
		}
}
module.exports = Activations
