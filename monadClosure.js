// Monads are containers of functions that you can execute *later*.
// Those functions can take some (blocking) time to run.
// Once we call .preform() this function will be executed
// We can chain pure functions that "do" something practical

function createMonad(functionInput) {
  const functionContained = functionInput;

  const then = (functionChained) => {
    // Calculate result of contained function
    const resultFromContained = functionContained();

    // Create new monad, with function that we wish to add to chain
    // and pass it the result of previous function
    const newMonad = createMonad(() => {
      return functionChained(resultFromContained);
    });

    // Return that monad so we can chain more
    return newMonad;
  };

  const preform = () => {
    return functionContained();
  };

  return {
    then: then,
    preform: preform
  };
}

// Some pure calculation functions that take blocking time
const prepareCalculation = () => {
  console.log('Begin calculation...');
  sleep(1);
  return 'XxY';
};

const calculateValue = (firstPart) => {
  console.log('Calculating value...', firstPart);
  sleep(1);
  return firstPart + 'yZz';
};

const processValue = (unprocessed) => {
  console.log('Processing value...', unprocessed);
  sleep(1);
  return unprocessed.toUpperCase();
};

const outputValue = (processed) => {
  console.log('Outputting value...', processed);
  sleep(1);
  console.log('Done!');
};

// Blocking node sleep function
function sleep(n) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, n * 1000);
}

// Define monad
const main = createMonad(prepareCalculation).then(calculateValue).then(processValue).then(outputValue);

// You can chain functions as you wish
// const main = createMonad(prepareCalculation).then(outputValue);

// ... later in runtime

// The chain of functions "saved" inside main monad, is only executed when preform() is runned
// Run monad
main.preform();
