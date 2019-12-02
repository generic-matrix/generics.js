var cuda = require("/content/cuda-ts/src/index.js");
const device = cuda.getDevices()[0];
const context = cuda.createContext(device);
