var { Image } = require('image-js');

function make_mat(data){
  var res=[];
  var ctr=0;
  for(var i=0;i<Math.ceil(Math.sqrt(data.length))-1;i++){
    var row=[];
    for(var j=0;j<Math.ceil(Math.sqrt(data.length))-1;j++){
      row.push(data[ctr]);
      ctr++;
    }
    res.push(row);
  }
  return res;
}

function relu(arr){
    //max(0,num);
    for(var i=0;i<arr.length;i++){
        arr[i]=Math.max(0,arr[i]);
    }
    return arr;
}
function max_pooling(raw_arr,kernel_length){
    var arr=Array.from(raw_arr);
    Array.prototype.max = function() {
        return Math.max.apply(null, this);
    };
     var res=[];
     if(arr.length<kernel_length){
         throw new Error("kernel is bigger then the image ");
     }else{
           var i,j,temparray,chunk = kernel_length;
            for (i=0,j=arr.length; i<j; i+=chunk) {
                temparray = arr.slice(i,i+chunk);
                var pooled=Math.max.apply(null,temparray);
                res.push(pooled/255);
            }
         return res;
     }
}


async function image_flattenX(dir,img_length,img_height,kernel_size,conv_options,conv_kernel) {
    var flatten=[];
      var img = await Image.load(dir);
      var image=img.resize({ width:img_length,height:img_height});
      img=img.gray();
      if(conv_kernel!=null){
      image=image.convolution(conv_kernel,conv_options);
      }

      flatten.push(max_pooling(relu(image.data),kernel_size));

      try{
        process.stdout.cursorTo(0);
        process.stdout.write("Processed the image with dir: "+dir);
      }catch(err){
        console.info("Processed the image with dir: "+dir);
      }

    return flatten;


}

class Convolution{


async image_flatten (dir,img_length,img_height,kernel_size,conv_options,conv_kernel){
      return image_flattenX(dir,img_length,img_height,kernel_size,conv_options,conv_kernel);
  }

}

module.exports = Convolution;
