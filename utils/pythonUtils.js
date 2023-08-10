const { spawn } = require('child_process');

/**
 * @desc    Execute Python script
 * @param   {string} pythonScriptPath - Path of the Python script file
 * @param   {Object} args - Arguments to be passed to the `spawn` function
 * @returns {Promise<void>}
 */
const executePythonScript = (pythonScriptPath, args = []) => {
  console.log('Script start Running');

  const pythonExecutablePath = `${process.env.PYTHON_EXE_PATH_TF_TORCH}`;

  return new Promise((resolve, reject) => {
    const process = spawn(pythonExecutablePath, [pythonScriptPath, ...args]);

    process.stdout.on('data', (data) => {
      console.log(`Python script output: ${data}`);
    });

    process.stderr.on('data', (data) => {
      console.error(`Python script error: ${data}`);
    });

    process.on('close', (code) => {
      if (code === 0) {
        // Script execution successful
        resolve();
      } else {
        // Script execution failed
        console.log('Script execution failed for some reason');

        reject(new Error('Script execution failed'));
      }
    });
  });
};

module.exports = {
  executePythonScript,
};
