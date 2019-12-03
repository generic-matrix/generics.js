const csv = require('csv-parser');
const fs = require('fs');
const Convolution =require("./Convolution.js");
var json={};
function check_neglect(neglect_arr,obj){
	if(neglect_arr.includes(obj)){
		return true;
	}else{
		return false;
	}
}

function shuffleArray(array1,array2) {
	for (let i = array1.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[array1[i], array1[j]] = [array1[j], array1[i]];
		[array2[i], array2[j]] = [array2[j], array2[i]];
	}
}


function reduce(numberToReduce,limitNumber) {
	return numberToReduce/1000;
}
function get_max(d_arr){
	var max=-1;
	for(var i=0;i<d_arr.length;i++){
		var num=Math.max.apply(null,d_arr[i]);
		if(num>max){
			max=num;
		}
	}
	return max;
}

function divide_by_max(arr,max){
	for(var i=0;i<arr.length;i++){
		for(var j=0;j<arr[i].length;j++){
			arr[i][j]=arr[i][j]/max;
		}
	}
}

function hash_row(row,x_axis,y_axis,fill_type,fill_with){
	if(fill_type==1){
		//fill with the user give number
		for(var i=0;i<x_axis.length;i++){
			if(row[x_axis[i]]==null||row[x_axis[i]]==undefined || row[x_axis[i]].length==0){
				row[x_axis[i]]=fill_with["x_axis"][i];
			}
		}
		for(var i=0;i<y_axis.length;i++){
			if(row[y_axis[i]]==null||row[y_axis[i]]==undefined || row[y_axis[i]].length==0){
				row[y_axis[i]]=fill_with["y_axis"][i];
			}
		}

	}else if(fill_type==2){
		//fill with a random value..
		for(var i=0;i<x_axis.length;i++){
			if(row[x_axis[i]]==null||row[x_axis[i]]==undefined || row[x_axis[i]].length==0){
				row[x_axis[i]]=Math.random()*100;
			}
		}
		for(var i=0;i<y_axis.length;i++){
			if(row[y_axis[i]]==null||row[y_axis[i]]==undefined || row[y_axis[i]].length==0){
				row[y_axis[i]]=Math.random()*100;
			}
		}
	}else{
		if(fill_type!=0){
			throw new Error("Invalid fill type.");
		}
	}


	for(var i=0;i<x_axis.length;i++){
		//check with respect to fill type..
		if(json[x_axis[i]]==undefined && isNaN(Number(row[x_axis[i]]))==true){
			json[x_axis[i]]=[];
		}
		//check if available..
		if(!isNaN(Number(row[x_axis[i]]))==true){
			row[x_axis[i]]=Number(row[x_axis[i]]);
		}else{
			if(json[x_axis[i]].includes(row[x_axis[i]])){
				//change the value accoringly
				row[x_axis[i]]=json[x_axis[i]].indexOf(row[x_axis[i]]);
			}else{
				//add the value...
				json[x_axis[i]].push(row[x_axis[i]]);
				row[x_axis[i]]=(json[x_axis[i]].length-1);
			}
		}

	}

	for(var i=0;i<y_axis.length;i++){
		if(json[y_axis[i]]==undefined && isNaN(Number(row[y_axis[i]]))==true){
			json[y_axis[i]]=[];
		}
		//check if available..
		if(!isNaN(Number(row[y_axis[i]]))==true){
			row[y_axis[i]]=Number(row[y_axis[i]]);
		}else{
			if(json[y_axis[i]].indexOf(row[y_axis[i]])!=-1){
				//change the value accoringly
				row[y_axis[i]]=json[y_axis[i]].indexOf(row[y_axis[i]]);
			}else{
				//add the value...
				json[y_axis[i]].push(row[y_axis[i]]);
				row[y_axis[i]]=(json[y_axis[i]].length-1);
			}
		}
	}
	return row;
}


function image_pre_process(x_axis,y_axis,dir,img_length=500,img_height=500,kernel_size=2,conv_options="direct",conv_kernel=null,find_files,callback){
	var j_arr=[];
	var conv=new Convolution();
	find_files(dir).then(function(files){
		files.forEach(function(file){
			find_files(dir+file).then(function(images){
				images.forEach(function(class_label) {
					var obj={};
					obj["dir"]=dir+file+"/"+class_label;
					obj["label"]=file;
					j_arr.push(obj);
					if(files[files.length-1]==file && images[images.length-1]==class_label){
						j_arr.forEach(function(obj) {
							conv.image_flatten(obj["dir"],img_length,img_height,kernel_size,conv_options,conv_kernel).then(function(array) {
								if(array!=null){
									array.forEach(function(x) {
										x_axis.push(x);
										var y=[obj["label"]];
										y_axis.push(y);
										if(callback!=null){
											callback(x_axis.length);
										}
									});
								}else{
									throw new Error("Error to preprocess the files.");
								}
							});
						});
					}
				});
			});
		});
	});
}


function find_filesX(dir,threashold) {
	let result = []
	let files = fs.readdirSync(dir)
	result = files.splice(0,threashold);
	for (var i = 0; i < result.length;i++){
		result[i] = dir+result[i];
	}
	return result;
}

function get_sampling(x_axis,image_dir,i,threashold,dir,img_length,img_height,kernel_size,conv_options,conv_kernel) {
	var conv = new Convolution();
	return new Promise(function(resolve, reject){
		conv.image_flatten(image_dir,img_length,img_height,kernel_size,conv_options,conv_kernel).then(function (array) {
			array.forEach(function (x) {
				var lbl = keys[i];
				x_axis[lbl.toString()].push(x);
				resolve(i);
			});
		});
	});
}

function sub_sampling(x_axis,threashold,dir, img_length, img_height, kernel_size,conv_options,conv_kernel) {
	for (var i = 0; i < keys.length; i++) {
		var lbl = keys[i];
		x_axis[lbl.toString()] = [];
	}
	for (var i = 0; i < keys.length; i++) {

		var images = find_filesX(dir + keys[i] + "/", threashold);
		images.forEach(function (image_dir,idx,array) {
			get_sampling(x_axis,image_dir,i,threashold,dir,img_length,img_height,kernel_size,conv_options,conv_kernel).then(function (index) {
				console.info("\n Processed with key : " + keys[index]);
			});
		});
	}
}



function encoding(y_axis) {
	//get the unique values..
	if(typeof(y_axis[0][0])=="object"){
		throw new Error("Objects cannot be encoded..");
	}

	var arr=y_axis.map(item => item[0])
		.filter((value, index, self) => self.indexOf(value) === index)
	for(var i=0;i<y_axis.length;i++){
		var ar=[];
		ar[0]=arr.indexOf(y_axis[i][0])/(arr.length-1);
		y_axis[i]=ar;
	}
	var json={};
	json["y_axis"]=y_axis;
	json["key"]=arr;
	return json;
}
/**
* A class for all pre processing activities
*/
class Preprocessing{
     /**
    * To pre process the image.
    * @param {array}  x_axis : The x_axis will be filled as the image is getting processed .
    * @param {array}  y_axis : The y_axis  will be filled as the image is getting processed .
    * @param {string}  dir : The  directory where the model must be saved.
    * @param {number}  img_length : The length of the image.
    * @param {number}  img_height  : The height of the image.
    * @param {number}  kernel_size : The size of the maxpooling kernel.
    * @param {string}  conv_options : Refer : https://image-js.github.io/image-js/#imageconvolution
    * @param {array}  conv_kernel : The kernel which must be used to process the image.
    * @param {function}  find_files : A callback function which is needed to find images in a directory defined by a user.
    * @param {function}  callback : A callback function which takes a argument number_of_images_processed.
    */
	image_pre_process(x_axis,y_axis,dir,img_length=500,img_height=500,kernel_size=2,conv_options="direct",conv_kernel=null,find_files=null,callback=null){
		if(find_files==null){
			throw new Error("The function to find files (find_files) is not passed.Please do add a function with your constraint needed.");
		}
		return image_pre_process(x_axis,y_axis,dir,img_length,img_height,kernel_size,conv_options,conv_kernel,find_files,callback);
	}
    /**
    * A class for label encoding .
    * @param {array}  y_axis : The y_axis is the output  for which we need the predictions.
    * @returns {array}  result : The result can be used to classify the data.
    */
    async label_encoding(y_axis){
		return encoding(y_axis);
	}
    /**
    * To get the samples of images for qualy check and data visualization or other purposes.
    * @param {JSON}  x_axis : A empty JSON which will be filled as the image is getting processed.
    * @param {number}  threashold : The threashold is the sample images taken from each class.
    * @param {string}  dir : The  directory where the model must be saved.
    * @param {number}  img_length : The length of the image.
    * @param {number}  img_height  : The height of the image.
    * @param {number}  kernel_size : The size of the maxpooling kernel.
    * @param {string}  conv_options : Refer : https://image-js.github.io/image-js/#imageconvolution
    * @param {array}  conv_kernel : The kernel which must be used to process the image.
    */
	sub_sampling(x_axis,threashold,dir, img_length, img_height, kernel_size,conv_options,conv_kernel){
		return sub_sampling(x_axis,threashold,dir, img_length, img_height, kernel_size,conv_options,conv_kernel);
	}

    /**
    * To parse the csv and convet it to JSON .
    * @param {string}  dir : A empty JSON which will be filled as the image is getting processed.
    * @param {number}  fill_type : 
    *       If fill_type==0 : Reject the row .
    *       If fill_type==1 :  Fill with the user give number which is given in fill_json (the last param)
    *       If fill_type==2 :  If fill with random values.
    * @param {array}  x_axis : The  directory where the model must be saved.
    * @param {array}  y_axis : The length of the image.
    * @param {number}  maximum_val  : The height of the image.
    * @param {JSON}  fill_json :  The Array you to fill with (will be applicable if fill_type==1 will be selected.)
            Example 
                var fill_json={
                "x_axis":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                "y_axis":[0]
                };
    */
	async parse_csv(dir,fill_type,x_axis,y_axis,maximum_val,fill_json=null){
		return new Promise(function(resolve, reject) {
			json={};
			var y=[];
			var x=[];
			var row_length=-1;
			fs.createReadStream(dir)
				.pipe(csv())
				.on('data', (row) => {
					if(row_length==-1){
						row_length=Object.keys(row).length;
					}
					if(Object.keys(row).length!=row_length){
						throw new Error("The rows length does not match in the csv file.");
					}
					var flag=1;
					if(fill_type==0){
						//reject the row..
						for(var i=0;i<x_axis.length;i++){
							if(row[x_axis[i]]==null||row[x_axis[i]]==undefined || row[x_axis[i]].length==0){
								flag=0;
								break;
							}
						}
						for(var i=0;i<y_axis.length;i++){
							if(row[y_axis[i]]==null||row[y_axis[i]]==undefined || row[y_axis[i]].length==0){
								flag=0;
								break;
							}
						}
					}
					//hash y axis
					//hash x axis..
					if(flag==1){
						var arr=[];
						row=hash_row(row,x_axis,y_axis,fill_type,fill_json);
						y_axis.forEach(function(obj){
							arr.push(row[obj]);
						});
						y.push(arr);

						arr=[];
						x_axis.forEach(function(obj){
							arr.push(row[obj]);
						});
						x.push(arr);
					}

				})
				.on('end', () => {
					if(maximum_val==undefined){
						var max=get_max(x);
						var max2=get_max(y);
						if(max<max2){
							max=max2;
						}
						if(max!=0){
							divide_by_max(x,max);
							divide_by_max(y,max);
						}
						maximum_val=max;
					}else{
						divide_by_max(x,maximum_val);
						divide_by_max(y,maximum_val);
					}
					//shuffle the elements..
					shuffleArray(x,y);
					console.log('CSV file successfully processed');
					json["x_axis"]=x;
					json["y_axis"]=y;
					json["max_val"]=maximum_val;
					resolve(json);
				});
		});
	}
}
module.exports = Preprocessing
