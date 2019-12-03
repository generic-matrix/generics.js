/**
 * generics.js
 * https://trygistify.com/generics
 *
 * A minimal Deep learning library for the web.
 *
 * @version 0.1.0
 * @date Tue Dec 03 2019 09:51:10 GMT-0500 (Eastern Standard Time)
 *
 * @license MIT
 * The MIT License
 *
 * Copyright (c) 2019 generics.js team
 */

'use strict'
const  Network =require("./src/ANN.js");
const  Neuron =require("./src/Neuron.js");
const  Utilities =require("./src/Utilities.js");
const Activation =require("./src/Activation.js");
const  Pre_Processing =require("./src/Pre_processing.js");
const Convolution =require("./src/Convolution.js");
const  ANN =require("./src/ANN.js");

module.exports ={
  Network,
  Neuron,
  Utilities,
  Activation,
  Pre_Processing,
  Convolution,
  ANN
};
