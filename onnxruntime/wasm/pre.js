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



function asmjsMangle(x) {
  var unmangledSymbols = ["stackAlloc", "stackSave", "stackRestore"];
  return x.indexOf("dynCall_") == 0 || unmangledSymbols.includes(x) ? x : "_" + x;
}

function exportAsmFunctions(asm) {
  var global_object = this;
  for (var __exportedFunc in asm) {
    var jsname = asmjsMangle(__exportedFunc);
    Module[jsname] = asm[__exportedFunc];
    if (global_object) {
      global_object[__exportedFunc] = asm[__exportedFunc];
    }
  }
}


function fallbackGemm(gemmToFallbackFunctionsMap) {
  // The fallback gemm implementation
  const FALLBACK_GEMM = "asm";

  let fallbackGemmModuleExports = {};
  for (let key in gemmToFallbackFunctionsMap) {
    fallbackGemmModuleExports[key] = (...a) =>
      Module[FALLBACK_GEMM][gemmToFallbackFunctionsMap[key]](...a);
  }
  return fallbackGemmModuleExports;
}

/**
* Custom call to instantiate WebAssembly module. so we can use custom imports 
*/ Module["instantiateWasm"] = async (info, receiveInstance) => {
  const wasmBinaryFile = findWasmBinary();
  const bytes = await getBinaryPromise(wasmBinaryFile);
  const module = await WebAssembly.compile(bytes);
  let imports = getWasmImports();

  // XXX mozIntGemm can't be used from web pages - we use a fallback if we are not privileged 
  const OPTIMIZED_GEMM = "mozIntGemm";

  const optimizedGemmModule = WebAssembly[OPTIMIZED_GEMM];
  if (!optimizedGemmModule) {
    const GEMM_TO_FALLBACK_FUNCTIONS_MAP = {
      int8_prepare_a: "int8PrepareAFallback",
      int8_prepare_b: "int8PrepareBFallback",
      int8_prepare_b_from_transposed: "int8PrepareBFromTransposedFallback",
      int8_prepare_b_from_quantized_transposed:
        "int8PrepareBFromQuantizedTransposedFallback",
      int8_prepare_bias: "int8PrepareBiasFallback",
      int8_multiply_and_add_bias: "int8MultiplyAndAddBiasFallback",
      int8_select_columns_of_b: "int8SelectColumnsOfBFallback",
    };
    imports.wasm_gemm = fallbackGemm(GEMM_TO_FALLBACK_FUNCTIONS_MAP);
  }

  else {
    var INITIAL_MEMORY = 16777216;
    var gemmWasmMemory = new WebAssembly.Memory({
      "initial": INITIAL_MEMORY / 65536,
      "maximum": 65536, // Maximum number of pages (4 GB)
      "shared": true
    });
    const optimizedGemmModuleExports = new WebAssembly.Instance(optimizedGemmModule(), {
      "": {
        memory: gemmWasmMemory
      }
    }).exports;
    imports.wasm_gemm = optimizedGemmModuleExports;
  }
  function mozReceiveInstance(instance) {
    // XXX do we need a moz specific stuff here?
    //var exports = instance.exports;
    //Module.asm = exports;
    // wasmTable = Module.asm.__indirect_function_table; ???
    //exportAsmFunctions(exports);
    return receiveInstance(instance);
  }
  try {
    var instance = new WebAssembly.Instance(module, imports);
    mozReceiveInstance(instance);
  } catch (error) {
    console.error("Error creating WebAssembly instance:", error);
    throw error;
  }
};

