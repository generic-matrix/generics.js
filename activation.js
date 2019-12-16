
let sigmoid=function(A){
    if(A!=null && A.length==0 ){
        throw new Error("Invalid inputs.");
    }else if(typeof(A[0])=="number"){
        //linear..
        for(let i=0;i<A.length;i++){
            A[i]=1 / (1 + Math.exp(A[i] * -1.0));
        }
    }else{
        //matrix..
        for(let i=0;i<A.length;i++){
            for(let j=0;j<A[0].length;j++){
                A[i][j]=1 / (1 + Math.exp(A[i][j] * -1.0));
            }
        }
    }
    return A;
}


let tanh=function(A){
    if(A!=null && A.length==0 ){
        throw new Error("Invalid inputs.");
    }else if(typeof(A[0])=="number"){
        //linear..
        for(let i=0;i<A.length;i++){
            A[i]=(2/(1+Math.exp(-2*A[i]))) -1 ;
        }
    }else{
        //matrix..
        for(let i=0;i<A.length;i++){
            for(let j=0;j<A[0].length;j++){
                A[i][j]=(2/(1+Math.exp(-2*A[i][j]))) -1;
            }
        }
    }
    return A;
}


let sigmoid_derivative=function(A){
    if(A!=null && A.length==0 ){
        throw new Error("Invalid inputs.");
    }else if(typeof(A[0])=="number"){
        //linear..
        for(let i=0;i<A.length;i++){
            A[i]=A[i]*(1.00-A[i]);
        }
    }else{
        //matrix..
        for(let i=0;i<A.length;i++){
            for(let j=0;j<A[0].length;j++){
                A[i][j]=A[i][j]*(1.00-A[i][j]);
            }
        }
    }
    return A;
}

let tanh_derivative=function(A){
    if(A!=null && A.length==0 ){
        throw new Error("Invalid inputs.");
    }else if(typeof(A[0])=="number"){
        //linear..
        for(let i=0;i<A.length;i++){
            A[i]=(1-((2/(1+Math.exp(-2*A[i]))) -1));
        }
    }else{
        //matrix..
        for(let i=0;i<A.length;i++){
            for(let j=0;j<A[0].length;j++){
                A[i][j]=(1-((2/(1+Math.exp(-2*A[i][j]))) -1));
            }
        }
    }
    return A;
}

module.exports = {
    sigmoid:sigmoid,
    tanh:tanh,
    sigmoid_derivative:sigmoid_derivative,
    tanh_derivative:tanh_derivative
};
