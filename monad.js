// Monads are containers of functions that you can execute *later*.
// Those functions can take some (blocking) time to run.
// Once we call .preform() this function will be executed
class Monad {
  constructor(fn) {
    this.fn = fn;
  }

  flatMap(fn) {
    return new Monad(() => {
      fn(this.fn());
    });
  }

  preform() {
    return this.fn();
  }
}

const a = new Monad(() => {
  console.log('Begin calculation...');
  sleep(1);
  const calculated = 'XXyyZZ';
  console.log('Done!');
  return calculated;
});

const b = a.flatMap((calculated) => {
  console.log('Uppercasing...');
  sleep(1);
  uppercased = calculated.toUpperCase();
  console.log(uppercased)
});

// Later in runtime

b.preform();


// Blocking node sleep function
function sleep(n) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, n * 1000);
}
