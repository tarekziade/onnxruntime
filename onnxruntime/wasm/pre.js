// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

//
// This file contains the pre-run code for the ORT WebAssembly module. The code in this file will be injected into the
// final module using Emscripten's `--pre-js` option.


/**
 * Mount external data files of a model to an internal map, which will be used during session initialization.
 *
 * @param {string} externalDataFilesPath
 * @param {Uint8Array} externalDataFilesData
 */
Module['mountExternalData'] = (externalDataFilePath, externalDataFileData) => {
  if (externalDataFilePath.startsWith('./')) {
    externalDataFilePath = externalDataFilePath.substring(2);
  }
  const files = Module.MountedFiles || (Module.MountedFiles = new Map());
  files.set(externalDataFilePath, externalDataFileData);
};

/**
 * Unmount external data files of a model.
 */
Module['unmountExternalData'] = () => {
  delete Module.MountedFiles;
};

/**
 * A workaround for SharedArrayBuffer when it is not available in the current context.
 *
 * We need this workaround because Emscripten generates code that assumes `SharedArrayBuffer` is always available and
 * uses SharedArrayBuffer in this way:
 * ```js
 * buffer instanceof SharedArrayBuffer
 * ```
 *
 * This code will throw an error when SharedArrayBuffer is not available. Fortunately, we can use `WebAssembly.Memory`
 * to create an instance of SharedArrayBuffer even when SharedArrayBuffer is not available in `globalThis`.
 *
 * While this workaround allows the WebAssembly module to be loaded, it does not provide multi-threading features when
 * SharedArrayBuffer is not available in `globalThis`. The WebAssembly module will run well in a single thread, when:
 * - Module['numThreads'] is set to 1, and
 * - _OrtInit() is called with numThreads = 1.
 *
 * @suppress {checkVars}
 */
var SharedArrayBuffer = globalThis.SharedArrayBuffer ??
  new WebAssembly.Memory({ 'initial': 0, 'maximum': 0, 'shared': true }).buffer.constructor;


/**
* Custom call to instantiate WebAssembly module. so we can use custom imports 
*/ Module["instantiateWasm"] = async (info, receiveInstance) => {
  const wasmBinaryFile = findWasmBinary();
  const bytes = await getBinaryPromise(wasmBinaryFile);
  const module = await WebAssembly.compile(bytes);
  let imports = getWasmImports();

  const OPTIMIZED_GEMM = "mozIntGemm";
  const optimizedGemmModule = WebAssembly[OPTIMIZED_GEMM];
  const optimizedGemmModuleExports = new WebAssembly.Instance(optimizedGemmModule(), {
    "": {
      memory: wasmMemory
    }
  }).exports;

  imports.wasm_gemm = optimizedGemmModuleExports;

  try {
    var instance = new WebAssembly.Instance(module, imports);
    receiveInstance(instance);
  } catch (error) {
    console.error("Error creating WebAssembly instance:", error);
    throw error;
  }
};

